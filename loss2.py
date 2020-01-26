import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=torch.cuda.FloatTensor([1.0405, 0.9311, 1.0038, 1.1441, 1.0039, 0.8766])):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)
        self.dice = DiceLoss()
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)
        loss2 = self.dice(score, target)
        return loss + loss2

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.9, 
        min_kept=131072, weight=torch.cuda.FloatTensor([1.1405, 0.9311, 1.0038, 1.1941, 1.0039, 0.8566])): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none')
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        loss2 = pixel_losses.clone()
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean() + loss2

class OhemCrossEntropy2(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.9, 
        min_kept=131072, weight= torch.cuda.FloatTensor([1.1931, 0.9287, 0.9954, 1.1389, 1.0004, 0.8735])): 
        super(OhemCrossEntropy2, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.DiceLoss = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none')
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.nlll=  nn.NLLLoss()

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        loss2 = self.nlll(output, target)
        return loss + loss2

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=-1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight= torch.cuda.FloatTensor([1.0405, 0.9311, 1.0038, 1.1441, 1.0039, 0.8766]), reduction=reduction, ignore_index=ignore_index)
        self.focal = FocalLoss()
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        focal_loss = self.focal(output, target)
        return 3 * CE_loss + dice_loss + 2 * focal_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=-1):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
    