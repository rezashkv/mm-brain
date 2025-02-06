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
