import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
writer = SummaryWriter()
class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
        
        dist = torch.rand(x.shape).sub(0.5).to(x.device)
        dist = l2_norm(dist)
        for _ in range(self.vat_iter):
            dist.requires_grad_()
            adv_examples = x + self.xi * dist
            adv_pred = model(adv_examples)
            adv_logp = F.log_softmax(adv_pred, dim=1)
            adv_distance = F.kl_div(adv_logp, pred, reduction='batchmean')
            if adv_distance <= 0 :
              print("<=0 value. Stop training!")
            adv_distance.backward()
            d_grad_l2 = l2_norm(dist.grad)
            model.zero_grad()   
        #calculate lds
        # grid_x = torchvision.utils.make_grid(x)
        # writer.add_image('original_image', grid_x, 0)
        r_adv = d_grad_l2 * self.eps
        # grid_adv = torchvision.utils.make_grid(x + r_adv)
        # writer.add_image('perturbed_image', grid_adv, 1)
        fin_adv_pred = model(x + r_adv)
        fin_adv_logp = F.log_softmax(fin_adv_pred, dim=1)
        lds = F.kl_div(fin_adv_logp, pred, reduction='batchmean')

        return lds    
        #raise NotImplementedError

def l2_norm(d):
    d_temp = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_temp, dim=1, keepdim=True) + 1e-8
    return d   