import os
import copy
import logging
import numpy as np
import sklearn
import torch
import delu as zero
from sklearn.model_selection import KFold
from training.training_classes import PretrainTeacher, TeacherStudentAdvML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data_dicts(datamodule, task='regression', train_size=0.8, random_seed=43):
    X_all = datamodule.dataset[datamodule.all_features].astype('float32').values
    if task == 'regression':
        y_all = datamodule.dataset[datamodule.target].astype('float32').values
    else:
        encoder = LabelEncoder()
        y_all = encoder.fit_transform(datamodule.dataset[datamodule.target].values)
    X = dict()
    y = dict()

    X['train'], X['test'], y['train'], y['test'] = train_test_split(
        X_all, y_all, train_size=train_size, random_state=random_seed
    )
    return X, y


def train(model_class, datamodule, args, log_function):
    X, y = prepare_data_dicts(datamodule, task=args.task, train_size=args.data_split_ratio[0],
                              random_seed=args.random_seed)

    X_train = X['train']
    y_train = y['train']
    X_test = X['test']
    y_test = y['test']

    splits = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.random_seed)
    best_scores = []
    if args.task == 'classification':
        best_f1s, best_prs, best_rcs, best_accs = [], [], [], []
    else:
        best_f1s, best_prs, best_rcs, best_accs = None, None, None, None
    checkpoint = None

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(y['train'].shape[0]))):
        model = model_class(args)
        model.to(args.device)

        X['train'], X['val'], X['test'], y['train'], y['val'], y['test'] = X_train[train_idx], X_train[val_idx], \
            X_test, \
            y_train[train_idx], y_train[val_idx], y_test

        y_std = y['train'].std().item()

        X_train_mri = X['train'][:, :datamodule.t1_data_num_features]
        X_train_snp = X['train'][:, datamodule.t1_data_num_features:]
        X_val_mri = X['val'][:, :datamodule.t1_data_num_features]
        X_val_snp = X['val'][:, datamodule.t1_data_num_features:]
        X_test_mri = X['test'][:, :datamodule.t1_data_num_features]
        X_test_snp = X['test'][:, datamodule.t1_data_num_features:]

        preprocess = sklearn.preprocessing.StandardScaler()
        preprocess.fit(X_train_mri)

        X['train'] = np.concatenate((preprocess.transform(X_train_mri), X_train_snp), axis=1)
        X['val'] = np.concatenate((preprocess.transform(X_val_mri), X_val_snp), axis=1)
        X['test'] = np.concatenate((preprocess.transform(X_test_mri), X_test_snp), axis=1)

        X = {
            k: torch.tensor(v, device=args.device)
            for k, v in X.items()
        }
        y = {k: torch.tensor(v, device=args.device) for k, v in y.items()}

        train_loader = zero.data.IndexLoader(len(X['train']), args.batch_size_train, device=args.device)

        progress = zero.ProgressTracker(patience=args.early_stopping_patience)
        # progress2 = zero.ProgressTracker(patience=args.early_stopping_patience)

        checkpoint = {'model': None, 'val': {'loss': torch.inf}, 'test': {'loss': torch.inf}}
        epoch, iteration = 0, 0
        best_f1, best_pr, best_acc, best_rc, best_r2 = 0, 0, 0, 0, 0
        if not args.debug:
            for epoch in range(1, args.num_epochs + 1):
                epoch_loss, total = 0., 0.
                for itr, batch_idx in enumerate(train_loader):
                    x_batch = X['train'][batch_idx]
                    y_batch = y['train'][batch_idx]
                    loss_dict = model.training_step(sample=x_batch,
                                                    target=y_batch,
                                                    y_std=y_std,
                                                    fold=fold,
                                                    iteration=iteration)
                    epoch_loss += loss_dict['loss'] * y_batch.shape[0]
                    total += y_batch.shape[0]
                    iteration += 1
                    if (itr + 1) % args.num_test_epochs == 0:
                        eval_val = model.eval_model(X, y, 'val', y_std)
                        eval_test = model.eval_model(X, y, 'test', y_std)
                        model.add_eval_results_to_writer(eval_val, 'Val', fold=fold, iteration=iteration)
                        model.add_eval_results_to_writer(eval_test, 'Test', fold=fold, iteration=iteration)
                        logging.warning(log_function(loss=loss_dict['loss'].item(), eval_val=eval_val,
                                                     eval_test=eval_test, fold=fold, epoch=epoch,
                                                     num_epochs=args.num_epochs, itr=itr + 1,
                                                     num_iter=len(train_loader)))
                        checkpoint = update_checkpoint(checkpoint, model, eval_val, eval_test, args, fold,
                                                       iteration)
                        log_best_checkpoint_dict(checkpoint)
                    else:
                        logging.warning(log_function(loss_dict['loss'].item(), epoch=epoch, fold=fold,
                                                     num_epochs=args.num_epochs, itr=itr + 1,
                                                     num_iter=len(train_loader)))

                # Testing at the end of epoch.
                epoch_loss = epoch_loss / total
                logging.warning('Evaluating Network at the end of epoch #{}:'.format(epoch))
                eval_val = model.eval_model(X, y, 'val', y_std=y_std)
                eval_test = model.eval_model(X, y, 'test', y_std=y_std)
                model.lr_scheduler_step(eval_val['loss'])
                model.add_eval_results_to_writer(eval_val, 'Val', fold=fold, epoch=epoch)
                model.add_eval_results_to_writer(eval_test, 'Test', fold=fold, epoch=epoch)
                args.writer.add_scalar('Train/loss_Epoch', epoch_loss, epoch)
                logging.warning(log_function(epoch_loss, eval_val, eval_test, fold=fold, epoch=epoch,
                                             num_epochs=args.num_epochs))
                checkpoint = update_checkpoint(checkpoint, model, eval_val, eval_test, args, fold, iteration)

                if args.task == 'classification':
                    progress.update(1 * eval_val['f1'])
                else:
                    progress.update(-1 * eval_val['loss'])
                if progress.success:
                    if args.task == 'classification':
                        best_f1 = eval_val['f1']
                        best_pr = eval_val['precision']
                        best_rc = eval_val['recall']
                        best_acc = eval_val['accuracy']
                    else:
                        best_r2 = eval_test['r2']
                    print(' <<< BEST VALIDATION EPOCH', end='')
                    logging.warning('BEST VALIDATION EPOCH')
                if progress.fail:
                    logging.warning(
                        'Early Stopping after {} epochs without improvement'.format(args.early_stopping_patience))
                    break

        else:
            model.train()
            train_batch = next(iter(train_loader))
            for epoch in range(1, args.num_epochs + 1):
                model.train()
                x_batch = train_batch[0]
                y_batch = train_batch[1]
                loss_dict = model.training_step(sample=x_batch,
                                                target=y_batch,
                                                y_std=y_std,
                                                fold=fold,
                                                iteration=iteration)
                epoch_loss = loss_dict['loss']
                eval_val = model.eval_model(X, y, 'val', y_std=y_std)
                eval_test = model.eval_model(X, y, 'test', y_std=y_std)
                model.add_eval_results_to_writer(eval_val, 'Val', fold=fold, epoch=epoch)
                model.add_eval_results_to_writer(eval_test, 'Test', fold=fold, epoch=epoch)
                logging.warning(log_function(epoch_loss.item(), eval_val, eval_test, fold, epoch,
                                             args.num_epochs))
                checkpoint = update_checkpoint(checkpoint, model, eval_val, eval_test, args, fold, epoch)

        best_scores += [best_r2]
        if args.task == 'classification':
            best_rcs += [best_rc]
            best_prs += [best_pr]
            best_accs += [best_acc]
            best_f1s += [best_f1]

        checkpoint_class_name = 'class_checkpoint_fold_{}_iter_{}.pth'.format(fold, checkpoint['iter'])
        file_name_class = os.path.join(args.checkpoint_dir, checkpoint_class_name)
        logging.warning('Saving Checkpoint: {}'.format(file_name_class))
        torch.save(checkpoint, file_name_class)

    return checkpoint, best_scores, best_accs, best_prs, best_rcs, best_f1s


def pretrain_teacher_model(datamodule, args):
    checkpoint, best_scores, _, _, _, _ = train(PretrainTeacher, datamodule, args, prepare_logging_message)
    log_best_checkpoint_dict(checkpoint)
    return checkpoint, best_scores, None, None, None, None


def train_teacher_student_adv_ml(datamodule, args):
    checkpoint, best_scores, best_accuracies, best_precisions, best_recalls, best_f1s = train(TeacherStudentAdvML,
                                                                                              datamodule, args,
                                                                                              prepare_logging_message_adv_ml)
    log_best_checkpoint_dict(checkpoint)
    return checkpoint, best_scores, best_accuracies, best_precisions, best_recalls, best_f1s


def prepare_logging_message_adv_ml(loss, eval_val=None, eval_test=None, fold=None, epoch=None, num_epochs=None,
                                   itr=None, num_iter=None):
    # items_to_log = ['class_loss_s', 'distill_loss_s', 'loss_D_fake', 'loss_D_real', 'gan_teacher_loss']
    items_to_log = ['class_loss_s', 'distill_loss_s', 'loss_D_GAN', 'loss_D_radius', 'loss_G_GAN']
    if eval_val is None:  # At the end of iter without evaluating on val/test set.
        log = 'Fold:{} Epoch: {}/{} \t Iter: {}/{}'.format(fold, epoch, num_epochs, itr, num_iter)
        for k, v in loss.items():
            if k in items_to_log:
                log += ' \t {}: {:.4f}'.format(k, v)
    else:
        if itr is None:
            log = 'Fold:{} Epoch: {}/{}\n'.format(fold, epoch, num_epochs)
            for k, v in eval_val.items():
                log += 'val_{}: {:.4f}\n'.format(k, v)
            for k, v in eval_test.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
        else:
            log = 'Fold:{} Epoch: {}/{} \t Iter: {}/{}\n'.format(fold, epoch, num_epochs, itr, num_iter)
            for k, v in eval_val.items():
                log += 'val_{}: {:.4f}\n'.format(k, v)
            for k, v in eval_test.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
    return log


def log_best_checkpoint(best_checkpoint_dict):
    logging.warning('Best Model so far:')
    for key, val in best_checkpoint_dict.items():
        if key == 'model':
            continue
        logging.warning('{}: {:.4f}'.format(key, val))


def update_checkpoint(curr_checkpoint_dict, model, eval_val, eval_test, args, fold, iteration):
    assert (args.metric == 'loss' or args.metric == 'auc' or args.metric == 'acc'), \
        'wrong metric: {}'.format(args.metric)

    if args.metric == 'loss':
        if eval_val['loss'] > curr_checkpoint_dict['val']['loss']:
            return curr_checkpoint_dict
    elif eval_val[args.metric] < curr_checkpoint_dict['val'][args.metric]:
        return curr_checkpoint_dict

    updated_dict = dict(curr_checkpoint_dict)
    updated_dict['model'] = copy.deepcopy(model.state_dict())
    updated_dict['val'] = dict(eval_val)
    updated_dict['test'] = dict(eval_test)
    updated_dict['fold'] = fold
    updated_dict['iter'] = iteration
    return updated_dict


def save_checkpoint(checkpoint, args):
    checkpoint_name = 'checkpoint_iter_{}.pth'.format(checkpoint['iter'])
    file_name = os.path.join(args.checkpoint_dir, checkpoint_name)
    logging.warning('Saving Checkpoint: {}'.format(file_name))
    torch.save(checkpoint, file_name)


def prepare_logging_message(loss, eval_val=None, eval_test=None, fold=None, epoch=None, num_epochs=None, itr=None,
                            num_iter=None):
    if eval_val is None:  # At the end of iter without evaluating on val/test set.
        log = 'Fold:{} \t Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f}'.format(fold, epoch, num_epochs, itr, num_iter,
                                                                              loss)

    else:
        if itr is None:
            log = 'Fold:{} \t Epoch: {}/{} \t Epoch_Loss: {:.4f}\n'.format(fold, epoch, num_epochs, loss)
            for k, v in eval_val.items():
                log += 'val_{}: {:.4f}\n'.format(k, v)
            for k, v in eval_test.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
        else:
            log = 'Fold:{} \t Epoch: {}/{} \t Iter: {}/{}\n'.format(fold, epoch, num_epochs, itr, num_iter)
            for k, v in eval_val.items():
                log += 'val_{}: {:.4f}\n'.format(k, v)
            for k, v in eval_test.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
    return log


def log_best_checkpoint_dict(best_checkpoint_dict):
    logging.warning('Best Model so far:')

    for key, val in best_checkpoint_dict.items():
        if key == 'model':
            continue

        elif key == 'val':
            logging.warning('Validation:')
            for k, v in val.items():
                logging.warning('{}: {:.4f}'.format(k, v))

        elif key == 'test':
            logging.warning('Test:')
            for k, v in val.items():
                logging.warning('{}: {:.4f}'.format(k, v))

        else:
            logging.warning('{}: {}'.format(key, val))


# mean:0.2927815331260712 std:0.06984631487274195