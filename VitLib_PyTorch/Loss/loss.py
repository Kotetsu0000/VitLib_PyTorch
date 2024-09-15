import torch
import torch.nn.functional as F
from torch import nn

#FNumberLoss, PrecisionLoss, RecallLossは完全自作
class FMeasureLoss(nn.Module):
    def __init__(self, Precision_weight:float=1.0, Recall_weight:float=1.0, in_sigmoid:bool=False) -> None:
        '''F-Measure Loss

        F値を模した損失関数. PrecisionとRecallの重みを調整できる.

        Args:
            Precision_weight (float): Precisionの重み. Defaults to 1.0.
            Recall_weight (float): Recallの重み. Defaults to 1.0.
            in_sigmoid (bool): 入力をSigmoid関数に通すか. Defaults to False.
        '''
        super(FMeasureLoss, self).__init__()
        self.Precision_weight = Precision_weight
        self.Recall_weight = Recall_weight
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets):
        if self.in_sigmoid:
            inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs_ = 1 - inputs
        targets_ = 1 - targets

        TP = (inputs * targets).sum()
        FP = (inputs_ * targets).sum()
        FN = (inputs * targets_).sum()

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)

        return 1 - (Precision * Recall * (self.Precision_weight + self.Recall_weight)) / (Precision * self.Precision_weight + Recall * self.Recall_weight)

class PrecisionLoss(nn.Module):
    def __init__(self, in_sigmoid:bool=False) -> None:
        '''Precision Loss

        適合率を模した損失関数.

        Args:
            in_sigmoid (bool): 入力をSigmoid関数に通すか. Defaults to False.
        '''
        super(PrecisionLoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets):
        if self.in_sigmoid:
            inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs_ = 1 - inputs
        targets_ = 1 - targets

        TP = (inputs * targets).sum()
        FP = (inputs_ * targets).sum()
        Precision = TP / (TP + FP)

        return 1 - Precision

class RecallLoss(nn.Module):
    def __init__(self, in_sigmoid:bool=False) -> None:
        '''Recall Loss
        
        再現率を模した損失関数.
        
        Args:
            in_sigmoid (bool): 入力をSigmoid関数に通すか. Defaults to False.
        '''
        super(RecallLoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets):
        if self.in_sigmoid:
            inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs_ = 1 - inputs
        targets_ = 1 - targets

        TP = (inputs * targets).sum()
        FN = (inputs * targets_).sum()
        Recall = TP / (TP + FN)
        return 1 - Recall

#以下の損失関数の参照元
#https://www.kaggle.com/code/parthdhameliya77/class-imbalance-weighted-binary-cross-entropy
class W_BCEWithLogitsLoss(torch.nn.Module):
    
    def __init__(self, w_p = None, w_n = None):
        super(W_BCEWithLogitsLoss, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, logits, labels, epsilon = 1e-7):
        
        ps = torch.sigmoid(logits.squeeze()) 
        
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss

#W_BCEWithLogitsLossを参考に自作
class W_BCELoss(torch.nn.Module):
    def __init__(self, w_p = None, w_n = None):
        super(W_BCELoss, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, logits, labels, epsilon = 1e-7):
        
        ps = logits.squeeze() 
        
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss

#以下の損失関数の参照元
#https://take-tech-engineer.com/pytorch-focal-loss/
class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, outputs, targets): 
      bce = self.bceloss(outputs, targets)
      bce_exp = torch.exp(-bce)
      focal_loss = (1-bce_exp)**self.gamma * bce
      return focal_loss.mean()

#DiceBCELoss, DiceLoss, IoULoss, FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLossの参照元
#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceBCELoss(nn.Module):
    def __init__(self, in_sigmoid:bool=False) -> None:
        super(DiceBCELoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class DiceLoss(nn.Module):
    def __init__(self, in_sigmoid:bool=False) -> None:
        super(DiceLoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class IoULoss(nn.Module):
    def __init__(self, in_sigmoid:bool=False) -> None:
        super(IoULoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
class ReverseIoULoss(nn.Module):
    def __init__(self, in_sigmoid:bool=False) -> None:
        super(ReverseIoULoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = 1 - inputs.view(-1)
        targets = 1 - targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    

class FocalLoss(nn.Module):
    ALPHA = 0.8
    GAMMA = 2
    def __init__(self, weight=None, size_average=True, in_sigmoid:bool=False) -> None:
        super(FocalLoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class TverskyLoss(nn.Module):
    ALPHA = 0.5
    BETA = 0.5
    def __init__(self, weight=None, size_average=True, in_sigmoid:bool=False) -> None:
        super(TverskyLoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    ALPHA = 0.5
    BETA = 0.5
    GAMMA = 1
    def __init__(self, weight=None, size_average=True, in_sigmoid:bool=False) -> None:
        super(FocalTverskyLoss, self).__init__()
        self.in_sigmoid = in_sigmoid

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.in_sigmoid:
            inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

class ComboLoss(nn.Module):
    ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
    BETA = 0.5
    CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (self.ALPHA * ((targets * torch.log(inputs)) + ((1 - self.ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.CE_RATIO * weighted_ce) - ((1 - self.CE_RATIO) * dice)
        
        return combo
    
#以下の損失関数の参照元
#https://github.com/jeya-maria-jose/KiU-Net-pytorch/blob/master/LiTS/loss/SS.py
class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # jaccard系数的定义
        s1 = ((pred - target).pow(2) * target).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target.sum(dim=1).sum(dim=1).sum(dim=1))

        s2 = ((pred - target).pow(2) * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target).sum(dim=1).sum(dim=1).sum(dim=1))

        # 返回的是jaccard距离
        return (0.05 * s1 + 0.95 * s2).mean()
