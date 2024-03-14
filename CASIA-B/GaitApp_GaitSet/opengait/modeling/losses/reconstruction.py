import torch
import torch.nn.functional as F
from .base import BaseLoss

class Recon_Loss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(Recon_Loss, self).__init__(loss_term_weight)
    
    def forward(self, source, target):
        
        loss = F.l1_loss(source, target)
        
        self.info.update({'loss': loss.detach().clone()})
        
        return loss, self.info