import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticFocalLoss(nn.Module):
    def __init__(self, gamma=0, weights=None, size_average=False):
        super(StaticFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.weights = weights
        if self.weights is not None:
            if len(self.weights.shape) == 1:
                self.weights = self.weights.unsqueeze(0)

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logits, 1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.weights is not None:
            weights = self.weights.repeat((logits.shape[0], 1))
            weights = weights.gather(1, target).view(-1)
            loss = loss * weights

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        loss = None
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class Huber(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(Huber, self).__init__()
        self.args = args
        self.reduction = reduction

    def forward(self, y_pred, y_target):
        diff = (torch.abs(y_pred - y_target)).to(self.args.device)
        flag = (diff < 1.).to(self.args.device)
        out = torch.zeros(y_pred.shape[0]).to(self.args.device)
        out[flag] = 0.5 * (diff[flag] ** 2)
        out[torch.logical_not(flag)] = diff[torch.logical_not(flag)] - 0.5

        if self.reduction is None:
            return out
        elif self.reduction == 'mean':
            return out.mean()



