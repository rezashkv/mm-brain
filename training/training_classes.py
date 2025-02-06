from collections import OrderedDict

import torch
import torch.nn as nn
import delu as zero
from torch.nn import init
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.multi_modal_models import MultiModalTeacher
from loss.regularization import Perturbation, Regularization
from loss.losses import GANLoss
from torch.optim import SGD, AdamW

from sklearn.metrics import f1_score,  precision_score, recall_score, balanced_accuracy_score, r2_score

from models.single_modal_models import StudentG, StudentG2, MLP, FTTransformer
import logging


class TrainingClass(nn.Module):
    """
        Parent of all training classes
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def eval_model(self, X, y, part, y_std):
        self.teacher.eval()
        prediction = []
        for batch in zero.iter_batches(X[part], 1024):
            prediction.append(self.apply_model(batch))
        prediction = torch.cat(prediction).squeeze(1)
        targets = y[part]

        if self.args.task == 'regression':
             # loss = self.loss_func(prediction, targets) ** 0.5 * y_std
             loss = self.loss_func(prediction, targets) ** 0.5
             r2 = r2_score(targets.cpu(), prediction.cpu())
             return {'loss': loss, 'r2': r2}
        else:
            loss = self.loss_func(prediction, targets)
            prediction = torch.argmax(prediction.cpu(), dim=1)
            targets = targets.cpu()
            f1 = f1_score(targets, prediction, average='macro')
            pr = precision_score(targets, prediction, average='macro')
            rc = recall_score(targets, prediction, average='macro')
            acc = balanced_accuracy_score(targets, prediction)
            return {'loss': loss, 'f1': f1, 'precision': pr, 'recall': rc, 'accuracy': acc}


    def add_train_losses_to_writer(self, loss_dict, fold, iteration):
        for k, v in loss_dict.items():
            self.args.writer.add_scalar('fold{}/Train/{}_Iter'.format(fold, k), v, iteration)

    def add_eval_results_to_writer(self, eval_results, partition, fold, iteration=None, epoch=None):
        if (iteration is None) and (epoch is None):
            raise ValueError('One of the iteration or epoch values should be not None.')
        if iteration is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('fold{}/{}/{}_Iter'.format(fold, partition, k), v, iteration)

        elif epoch is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('fold{}/{}/{}_Iter'.format(fold, partition, k), v, epoch)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class PretrainTeacher(TrainingClass):
    def __init__(self, args):
        super().__init__(args)

        self.x_img, self.x_gene = None, None
        assert args.teacher_snp_net_backbone in ['mlp', 'transformer'], "SNP Net Backbone has to be one of 'mlp' or" \
                                                                        " 'transformer'"
        if args.teacher_snp_net_backbone == 'mlp':
            snp_net_params = {'d_in': self.args.snp_num_features,
                              'd_layers': self.args.teacher_snp_net_d_layers,
                              'dropout': self.args.teacher_snp_net_dropout,
                              'd_out': self.args.teacher_snp_net_d_out,
                              'batch_norm': self.args.teacher_snp_net_batch_norm,
                              'activation': self.args.teacher_snp_net_activation}
        else:
            snp_net_params = {'n_num_features': self.args.snp_num_features,
                              'cat_cardinalities': None,
                              'n_blocks': self.args.teacher_snp_n_blocks,
                              'last_layer_query_idx': [-1],
                              'd_out': self.args.teacher_snp_net_d_out}
        if args.teacher_t1_net_backbone == 'mlp':
            t1_net_params = {'d_in': self.args.t1_num_features,
                             'd_layers': self.args.teacher_t1_net_d_layers,
                             'dropout': self.args.teacher_t1_net_dropout,
                             'd_out': self.args.teacher_t1_net_d_out,
                             'batch_norm': self.args.teacher_t1_net_batch_norm,
                             'activation': self.args.teacher_t1_net_activation}
        else:
            t1_net_params = {'n_num_features': self.args.t1_num_features,
                             'cat_cardinalities': None,
                             'n_blocks': self.args.teacher_t1_n_blocks,
                             'last_layer_query_idx': [-1],
                             'd_out': self.args.teacher_t1_net_d_out}

        self.teacher = MultiModalTeacher(t1_net_type=self.args.teacher_t1_net_backbone,
                                         t1_net_params=t1_net_params,
                                         snp_net_type=self.args.teacher_snp_net_backbone,
                                         snp_net_params=snp_net_params,
                                         output_dim=self.args.target_dim,
                                         task=self.args.task)

        if self.args.teacher_t1_optimizer == 'adamw':
            params = list(self.teacher.t1_net.parameters()) + list(self.teacher.regressor.parameters())
            self.t1_optimizer = AdamW(params, lr=self.args.t1_lr, weight_decay=self.args.teacher_t1_weight_decay,
                                      betas=self.args.betas)
        elif args.teacher_t1_optimizer == 'sgd':
            params = list(self.teacher.t1_net.parameters()) + list(self.teacher.regressor.parameters())
            self.t1_optimizer = SGD(params, lr=self.args.t1_lr, weight_decay=self.args.teacher_t1_weight_decay,
                                    momentum=self.args.momentum)
        else:
            raise NotImplementedError("Wrong optimizer specified")

        if self.args.teacher_snp_optimizer == 'adamw':
            self.snp_optimizer = AdamW(self.teacher.snp_net.parameters(), lr=self.args.snp_lr,
                                       weight_decay=self.args.teacher_snp_weight_decay, betas=self.args.betas)
        elif args.teacher_t1_optimizer == 'sgd':
            self.snp_optimizer = SGD(self.teacher.snp_net.parameters(), lr=self.args.snp_lr,
                                     weight_decay=self.args.teacher_snp_weight_decay, momentum=self.args.momentum)
        else:
            raise NotImplementedError("Wrong optimizer specified")

        if self.args.task == 'regression':
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = nn.NLLLoss()
        self.t1_lr_scheduler = ReduceLROnPlateau(self.t1_optimizer, patience=self.args.t1_lr_scheduler_patience,
                                                 verbose=self.args.lr_scheduler_verbose,
                                                 factor=self.args.t1_lr_scheduler_factor)
        self.snp_lr_scheduler = ReduceLROnPlateau(self.snp_optimizer, patience=self.args.snp_lr_scheduler_patience,
                                                  verbose=self.args.lr_scheduler_verbose,
                                                  factor=self.args.snp_lr_scheduler_factor)

    def training_step(self, sample, y_std, target, fold, iteration):
        self.teacher.train()
        self.t1_optimizer.zero_grad()
        self.snp_optimizer.zero_grad()

        y_pred = self.apply_model(sample)
        # loss = self.loss_func(y_pred.squeeze(1), target) ** 0.5 * y_std
        loss = self.loss_func(y_pred.squeeze(1), target) ** 0.5

        # ###################### Our regularization method #######################
        if self.args.functional_regularization:
            # expanded_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples)
            # expanded_logits = Perturbation.get_expanded_logits(y_pred, self.args.n_samples)
            expanded_logits = y_pred.repeat(1, self.args.n_samples).view(y_pred.shape[0] * self.args.n_samples, -1)
            # inf_image = Perturbation.perturb_tensor(x_img[1], self.args.n_samples)
            inf_gene = Perturbation.perturb_tensor(sample[:, self.teacher.t1_net.d_in:], self.args.n_samples,
                                                   perturbation_strength=self.args.perturbation_strength)
            x_im = sample[:, :self.teacher.t1_net.d_in]
            tens_dim = list(x_im.shape)
            x_img_rep = x_im.view(x_im.shape[0], -1)
            x_img_rep = x_img_rep.repeat(1, self.args.n_samples)
            x_img_rep = x_img_rep.view(x_img_rep.shape[0] * self.args.n_samples, -1)
            tens_dim[0] *= self.args.n_samples
            x_img_rep = x_img_rep.view(*tens_dim)

            # inf_output, _, _ = self.teacher(inf_image, inf_gene, (x_pheno.repeat(1, self.args.n_samples))
            # .view(x_pheno.shape[0] * self.args.n_samples, -1))
            inf_output, _, _ = self.teacher(x_img_rep, inf_gene)
            inf_loss = torch.nn.MSELoss()(inf_output, expanded_logits)

            # gradients = torch.autograd.grad(inf_loss, [inf_image, inf_gene], create_graph=True)
            gradients = torch.autograd.grad(inf_loss, [inf_gene], create_graph=True)
            # grads = [Regularization.get_batch_norm(gradients[k], loss=inf_loss,
            #                                        estimation=self.args.estimation) for k in range(2)]
            grads = [Regularization.get_batch_norm(gradients[k], loss=inf_loss,
                                                   estimation=self.args.estimation) for k in range(1)]

            inf_scores = torch.stack(grads)
            reg_term = Regularization.get_regularization_term(inf_scores, norm=self.args.norm,
                                                              optim_method=self.args.optim_method)

            loss += self.args.delta * reg_term

        #########################################################################

        # Lasso for the Gene Network
        if self.args.teacher_lasso != 0.:
            l1_loss = torch.tensor([0.]).to(self.args.device)
            for param in self.teacher.gene_net.parameters():
                l1_loss += self.args.teacher_gene_net_lasso * torch.sum(torch.abs(param))
            loss = loss + l1_loss

        if self.args.functional_regularization:
            loss_dict = {'loss': loss.detach(), 'func_reg': self.args.delta * reg_term.detach()}
        else:
            loss_dict = {'loss': loss.detach()}

        self.add_train_losses_to_writer(loss_dict, fold, iteration)
        loss.backward()
        self.t1_optimizer.step()
        self.snp_optimizer.step()
        if self.args.functional_regularization:
            self.snp_optimizer.zero_grad(set_to_none=True)
            self.t1_optimizer.zero_grad(set_to_none=True)
            del gradients

        return loss_dict

    def apply_model(self, x_num):
        if isinstance(self.teacher, MultiModalTeacher):
            out, self.x_img, self.x_gene = self.teacher(t1_data=x_num[:, :self.teacher.t1_net.d_in],
                                                        snp_data=x_num[:, self.teacher.t1_net.d_in:])
            return out
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(self.teacher)}.'
                ' Then you have to implement this branch first.'
            )

    def lr_scheduler_step(self, validation_metric):
        self.t1_lr_scheduler.step(validation_metric)
        self.snp_lr_scheduler.step(validation_metric)


class TeacherStudentAdvML(TrainingClass):
    def __init__(self, args, is_train=True):
        super().__init__(args)
        self.is_train = is_train
        # Models
        self.teacher = TeacherForAdvML(self.args, is_train=self.is_train)
        self.student = StudentForAdvML(self.args, is_train=self.is_train)
        self.netD = Discriminator(self.args, self.student.netG.hidden_dim)
        # Initialization for GAN
        self.init_weights()

        self.optim_st_class, self.optim_st_G = self.student.get_optimizer()
        self.optimizer_D = self.netD.get_optimizer()
        self.st_lr_scheduler = ReduceLROnPlateau(self.optim_st_class,
                                                 patience=self.args.student_class_lr_scheduler_patience,
                                                 verbose=self.args.lr_scheduler_verbose,
                                                 factor=self.args.student_class_lr_scheduler_factor)

        self.st_g_lr_scheduler = ReduceLROnPlateau(self.optim_st_G, patience=self.args.student_g_lr_scheduler_patience,
                                                   verbose=self.args.lr_scheduler_verbose,
                                                   factor=self.args.student_g_lr_scheduler_factor)

        self.discriminator_lr_scheduler = ReduceLROnPlateau(self.optimizer_D,
                                                            patience=self.args.discriminator_lr_scheduler_patience,
                                                            verbose=self.args.lr_scheduler_verbose,
                                                            factor=self.args.discriminator_lr_scheduler_factor)

        self.gan_loss = GANLoss(self.args.gan_mode)

        if self.args.task == 'regression':
            self.class_loss = nn.MSELoss()
        else:
            self.class_loss = nn.NLLLoss()

        self.mutual_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')

    def training_step(self, sample, y_std, target, fold, iteration):
        t1_sample = sample[:, :self.teacher.teacher.t1_net.d_in]
        snp_sample = sample[:, self.teacher.teacher.t1_net.d_in:]
        self.student.train()
        self.teacher.train()
        self.netD.train()

        y_pred_s, x_gene_fake = self.student(t1_sample, dropout=True)
        with torch.no_grad():
            y_pred_t, self.teacher.x_img, x_gene_real = self.teacher.teacher(t1_sample, snp_sample)
            self.teacher.x_gene = x_gene_real

        self.optim_st_class.zero_grad(), self.optim_st_G.zero_grad()
        # ####################### Classification and Mutual Loss #######################

        # ############################### Added August 2.
        # class_loss_s = self.class_loss(y_pred_s.squeeze(1), target) ** 0.5 * y_std
        class_loss_s = self.class_loss(y_pred_s.squeeze(1), target) ** 0.5
        if self.args.distill_confidence is not None:
            pred_probs = (torch.softmax(y_pred_t, 1)).gather(1, target.view(-1, 1)).squeeze()
            distill_loss_s = F.kl_div(
                F.log_softmax(y_pred_s[pred_probs > self.args.distill_confidence] / self.args.T, dim=1),
                F.softmax(y_pred_t[pred_probs > self.args.distill_confidence] / self.args.T, dim=1),
                reduction='batchmean') * self.args.T * self.args.T
        else:
            distill_loss_s = F.kl_div(F.log_softmax(y_pred_s / self.args.T, dim=1),
                                      F.softmax(y_pred_t / self.args.T, dim=1), reduction='batchmean') * self.args.T * \
                             self.args.T
        # ############################### Added August 2.

        # class_loss_s = self.class_loss(y_pred_s, y)
        # distill_loss_s = F.kl_div(F.log_softmax(y_pred_s / self.args.T, dim=1),
        #                           F.softmax(y_pred_t / self.args.T, dim=1), reduction='batchmean') * self.args.T * \
        #                  self.args.T

        # ####################### GAN model #######################
        # ####### update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero

        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_D = self.netD(x_gene_fake.detach())
        loss_D_fake = self.gan_loss(pred_fake_D, False)

        # Real
        pred_real = self.netD(x_gene_real)
        loss_D_real = self.gan_loss(pred_real, True)

        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        self.optimizer_D.step()  # update D's weights

        # ####### update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optim_st_G.zero_grad()  # set G's gradients to zero

        """Calculate GAN loss for the generator"""
        # G(A) should fake the discriminator
        pred_fake_G = self.netD(x_gene_fake)
        loss_G_GAN = self.gan_loss(pred_fake_G, True)
        # combine loss and calculate gradients
        # self.loss_G.backward()
        # self.optimizer_G.step()

        loss_s = class_loss_s + self.args.l1 * loss_G_GAN + self.args.l2 * distill_loss_s

        # ############################### Added August 2.
        gan_teacher_loss = 0.
        if self.args.l3 != 0.:
            y_pred_t_fake = self.teacher.forward_student_gene_pred(x_gene_fake)
            gan_teacher_loss = self.class_loss(y_pred_t_fake.squeeze(1), target)
            loss_s = loss_s + self.args.l3 * gan_teacher_loss
        # ############################### Added August 2.

        loss_s.backward()
        self.optim_st_G.step()
        self.optim_st_class.step()

        # ####################### Update Teacher Model #######################
        self.teacher.update_weights(self.student.t1_net.state_dict(), subnetwork='t1_net')
        self.teacher.update_weights(self.student.regressor.state_dict(), subnetwork='fc')

        # loss_dict = {'class_loss_s': class_loss_s, 'distill_loss_s': distill_loss_s,
        #              'loss_D_fake': loss_D_fake, 'loss_D_real': loss_D_real,
        #              'loss_D': loss_D, 'loss_s': loss_s}
        loss_dict = {'class_loss_s': class_loss_s, 'distill_loss_s': distill_loss_s,
                     'loss_D_fake': loss_D_fake, 'loss_D_real': loss_D_real,
                     'loss_D': loss_D, 'loss': loss_s, 'gan_teacher_loss': gan_teacher_loss}  # Added August 2.

        self.add_train_losses_to_writer(loss_dict, fold, iteration)
        return loss_dict

    def init_weights(self):
        """Initialize GAN weights.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        if self.args.init_type == 'default':
            return
        else:
            def init_func(m):  # define the initialization function
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if self.args.init_type == 'normal':
                        init.normal_(m.weight.data, 0.0, self.args.init_gain)
                    elif self.args.init_type == 'xavier':
                        init.xavier_normal_(m.weight.data, gain=self.args.init_gain)
                    elif self.args.init_type == 'kaiming':
                        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    elif self.args.init_type == 'orthogonal':
                        init.orthogonal_(m.weight.data, gain=self.args.init_gain)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % self.args.init_type)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)
                elif classname.find(
                        'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution
                    # applies.
                    init.normal_(m.weight.data, 1.0, self.args.init_gain)
                    init.constant_(m.bias.data, 0.0)

        logging.warning('#' * 20)
        logging.warning("initialize student's Generator with %s" % self.args.init_type)
        self.student.netG.apply(init_func)  # apply the initialization function <init_func>
        logging.warning("initialize Discriminator with %s" % self.args.init_type)
        self.netD.apply(init_func)

    def apply_teacher_model(self, x_num):
        if isinstance(self.teacher, MultiModalTeacher):
            return self.teacher(t1_data=x_num[:, :self.teacher.t1_net.d_in],
                                snp_data=x_num[:, self.teacher.t1_net.d_in:])
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(self.teacher)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def eval_model(self, X, y, part, y_std):

        self.student.eval()
        prediction = []
        for batch in zero.iter_batches(X[part], 1024):
            t1_sample = batch[:, :self.student.t1_net.d_in]
            prediction.append(self.student(t1_sample)[0])
        prediction = torch.cat(prediction).squeeze(1)

        targets = y[part]
        if self.args.task == 'regression':
            # loss = self.class_loss(prediction, targets) ** 0.5 * y_std
            loss = self.class_loss(prediction, targets) ** 0.5
            r2 = r2_score(targets.cpu(), prediction.cpu())
            return {'loss': loss, 'r2': r2}

        else:
            loss = self.class_loss(prediction, targets)
            prediction = torch.argmax(prediction.cpu(), dim=1)
            targets = targets.cpu()
            f1 = f1_score(targets, prediction, average='macro')
            pr = precision_score(targets, prediction, average='macro')
            rc = recall_score(targets, prediction, average='macro')
            acc = balanced_accuracy_score(targets, prediction)
            return {'loss': loss, 'f1': f1, 'precision': pr, 'recall': rc, 'accuracy': acc}

    def lr_scheduler_step(self, validation_metric):
        self.st_lr_scheduler.step(validation_metric)
        self.st_g_lr_scheduler.step(validation_metric)
        self.discriminator_lr_scheduler.step(validation_metric)


class StudentForAdvML(TrainingClass):
    def __init__(self, args, is_train=True):
        super().__init__(args)
        self.is_train = is_train
        self.t1_net, self.netG, self.regressor, self.clf = None, None, None, None
        self.build_network()

    def build_network(self):

        if self.args.teacher_t1_net_backbone == 'mlp':
            t1_net_params = {'d_in': self.args.t1_num_features,
                             'd_layers': self.args.teacher_t1_net_d_layers,
                             'dropout': self.args.teacher_t1_net_dropout,
                             'd_out': self.args.teacher_t1_net_d_out,
                             'batch_norm': self.args.teacher_t1_net_batch_norm,
                             'activation': self.args.teacher_t1_net_activation}
            self.t1_net = MLP.make_baseline(**t1_net_params)
        else:
            t1_net_params = {'n_num_features': self.args.t1_num_features,
                             'cat_cardinalities': None,
                             'n_blocks': self.args.teacher_t1_n_blocks,
                             'last_layer_query_idx': [-1],
                             'd_out': self.args.teacher_t1_net_d_out}
            self.t1_net = FTTransformer.make_default(**t1_net_params)
            self.t1_net.d_in = t1_net_params['n_num_features']
            self.t1_net.d_out = t1_net_params['d_out']
        if self.args.student_G_type == 1:
            self.netG = StudentG(self.args, self.t1_net.d_out, self.args.embed_dim_gene)
        elif self.args.student_G_type == 2:
            self.netG = StudentG2(self.args, self.t1_net.d_out, self.args.embed_dim_gene)
        else:
            raise ValueError

        self.regressor = nn.Linear(self.t1_net.d_out + self.netG.hidden_dim, self.args.target_dim)

        if self.args.task == 'classification':
            self.clf = nn.LogSoftmax(dim=1)

        if self.is_train:
            # Added August 17
            checkpoint = torch.load(self.args.pretr_teacher_dir, map_location='cpu')
            checkpoint_model = checkpoint['model']
            self.load_from_teacher_state_dict(checkpoint_model)
            # Added August 17

    def get_optimizer(self):
        params_c = list(self.t1_net.parameters()) + list(self.regressor.parameters())
        params_G = list(self.netG.parameters())

        if self.args.teacher_optimizer == 'adamw':
            optim_c = AdamW(params_c, lr=self.args.student_c_lr, weight_decay=self.args.student_c_weight_decay,
                            betas=self.args.student_c_betas)
            optim_G = AdamW(params_G, lr=self.args.student_G_lr, weight_decay=self.args.student_G_weight_decay,
                            betas=self.args.student_G_betas)
        elif self.args.teacher_optimizer == 'sgd':
            optim_c = SGD(params_c, lr=self.args.student_c_lr, weight_decay=self.args.student_c_weight_decay,
                          momentum=self.args.momentum)
            optim_G = SGD(params_G, lr=self.args.student_c_lr, weight_decay=self.args.student_c_weight_decay,
                          momentum=self.args.momentum)
        else:
            raise NotImplementedError("Unknown optimizer specified")

        return optim_c, optim_G

    def _load_from_checkpoint(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        checkpoint_model = checkpoint['model']
        if 'teacher' == (list(checkpoint_model.keys())[0])[:len('teacher')]:
            resnet_dict = OrderedDict()
            fc_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                name = k[8:]  # remove `module.`
                if 'resnet' in name:
                    resnet_dict[name[len('resnet') + 1:]] = v
                elif 'fc_pred' in name:
                    fc_dict[name[len('fc_pred') + 1:]] = v
        else:
            raise NotImplementedError
        self.resnet.load_state_dict(resnet_dict)
        self.regressor.load_state_dict(fc_dict)

    def load_from_teacher_state_dict(self, teacher_state_dict):
        if 'teacher' == (list(teacher_state_dict.keys())[0])[:len('teacher')]:
            t1_net_dict = OrderedDict()
            regressor_dict = OrderedDict()
            for k, v in teacher_state_dict.items():
                name = k[8:]  # remove `module.`
                if 't1_net' in name:
                    t1_net_dict[name[len('t1_net') + 1:]] = v
                elif 'regressor' in name:
                    regressor_dict[name[len('regressor') + 1:]] = v
        else:
            t1_net_dict, regressor_dict = OrderedDict(), OrderedDict()
            for k, v in teacher_state_dict.items():
                if 't1_net' in k:
                    t1_net_dict[k.replace('t1_net.', '')] = v
                elif 'regressor' in k:
                    regressor_dict[k.replace('regressor.', '')] = v
                else:
                    continue
        self.t1_net.load_state_dict(t1_net_dict)
        if self.args.student_c_load_from_teacher:
            self.regressor.load_state_dict(regressor_dict)

    def forward(self, x, dropout=True):
        t1_features = self.t1_net(x)
        # try:
        gene_rep_pred = self.netG(t1_features, dropout=dropout)
        # except Exception as e:
        #     print(e)
        #     import pdb
        #     pdb.set_trace()
        #     gene_rep_pred = t1_features

        if self.args.student_detach_gene:
            x = torch.cat([t1_features, gene_rep_pred.detach()], dim=1).type(torch.FloatTensor).to(self.args.device)
        else:
            x = torch.cat([t1_features, gene_rep_pred], dim=1).type(torch.FloatTensor).to(self.args.device)

        x = self.regressor(x)
        if self.clf:
            x = self.clf(x)

        return x, gene_rep_pred


class TeacherForAdvML(PretrainTeacher):
    def __init__(self, args, is_train=True):
        super(TeacherForAdvML, self).__init__(args)

        self.is_train = is_train

        if self.is_train:
            self._load_from_checkpoint(self.args.pretr_teacher_dir)

    def _load_from_checkpoint(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        checkpoint_model = checkpoint['model']
        if 'teacher' == (list(checkpoint_model.keys())[0])[:len('teacher')]:
            cls_state_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                name = k[8:]  # remove `teacher.`
                cls_state_dict[name] = v
            checkpoint_model = cls_state_dict
        self.teacher.load_state_dict(checkpoint_model)

    def update_weights(self, state_dic, subnetwork='t1_net'):
        """
        :param state_dic: State dict of the sub-network of the student.
        :param subnetwork: whether 'resnet' (CNN backbone) or 'fc' (classifier)
        """
        if subnetwork == 't1_net':
            curr_t1_dict_dict = self.teacher.t1_net.state_dict()
            new_t1_net_dict = OrderedDict()
            for k, v in state_dic.items():
                name = k
                new_t1_net_dict[name] = (1 - self.args.teacher_alpha) * v + \
                                        self.args.teacher_alpha * curr_t1_dict_dict[name]
            self.teacher.t1_net.load_state_dict(new_t1_net_dict)
        elif subnetwork == 'fc':
            curr_regressor_dict = self.teacher.regressor.state_dict()
            new_regressor_dict = OrderedDict()
            for k, v in state_dic.items():
                name = k
                new_regressor_dict[name] = (1 - self.args.teacher_alpha) * v + \
                                           self.args.teacher_alpha * curr_regressor_dict[name]
            self.teacher.regressor.load_state_dict(new_regressor_dict)

    def forward_student_gene_pred(self, student_gene_pred):
        self.set_requires_grad(self.teacher.regressor, False)
        x = self.teacher.regressor(
            torch.cat([self.x_img, student_gene_pred], dim=1).type(torch.FloatTensor).to(self.args.device))
        self.set_requires_grad(self.teacher.regressor, True)
        if self.args.task == 'classification':
            x = nn.LogSoftmax(dim=1)(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, args, input_dim):
        super(Discriminator, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 1)

    def get_optimizer(self):
        return AdamW(self.parameters(), lr=self.args.netD_lr, weight_decay=self.args.netD_weight_decay,
                     betas=self.args.netD_betas)

    def forward(self, x_input):
        x = self.fc1(x_input)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.fc3(x)
        return x
