 

import torch
import torch.nn as nn

# Part of the code comes from Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

class AMCLoss(nn.Module):
    def __init__(self,last_loss=torch.zeros(50,15), contrast_mode='all'):
        super(AMCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.last_loss = last_loss
        self.K = 15
    def forward(self, features, epoch, labels=None, mask=None, train=False):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # 同一个label为1 （bz, bz）
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #(bz * views, hidden) b1v1 b2v1..., b1v2 b2v2...
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) # (bz * views, bz * views)
            
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # all pairs


        mask = mask.repeat(anchor_count, contrast_count) #（bz * views, bz * views)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # The non diagonal line is positive pairs
        mask = mask * logits_mask 
        
        # compute log_prob 
        exp_logits = torch.exp(logits) * logits_mask # all case except self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mask_prob = mask * log_prob
        m_loss = []
        
        for i in range(contrast_count):
            for j in range(i+1, contrast_count):
                temp_l_1 = mask_prob[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size]
                temp_m_1 = mask[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size]
                
                temp_l_2 = mask_prob[j*batch_size:(j+1)*batch_size, i*batch_size:(i+1)*batch_size]
                temp_m_2 = mask[j*batch_size:(j+1)*batch_size, i*batch_size:(i+1)*batch_size]
                
                m_loss.append((temp_l_1.sum(1)+temp_l_2.sum(1)) / (temp_m_1.sum(1) +  temp_m_2.sum(1)))
        m_loss = torch.stack(m_loss)   # pair * batch_size    
        
        
        # Adaptive weight
        if epoch <= 2:
            w = torch.ones(15)
        else:
            w = self.last_loss[epoch - 1] / self.last_loss[epoch - 2]
        w = self.K * ((torch.exp(w) / torch.exp(w).sum())).cuda()
        now_loss = (torch.mul(w.clone().detach(), m_loss.T)).mean(dim=0)
        loss = - now_loss
        
        # Only updated during training
        if train == True:
            self.last_loss[epoch] = self.last_loss[epoch] + loss.detach().cpu()

        loss = loss.sum()
        return loss
