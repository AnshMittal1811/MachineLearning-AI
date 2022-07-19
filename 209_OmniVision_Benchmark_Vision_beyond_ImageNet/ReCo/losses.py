import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class PaCoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000,self_sup=False,sup=False):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.supt = supt
        self.num_classes = num_classes
        self.self_sup = self_sup
        self.sup = sup

    def forward(self, features, labels=None, sup_logits=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = ( features.shape[0] - self.K ) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)



        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        
        if not self.self_sup:
            anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)



        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask


        # add ground truth 

        if not self.self_sup and not self.sup:
            one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
            mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        # if self.sup:
        #     logits_mask = torch.ones(batch_size, self.num_classes).to(device)

        if not self.self_sup and not self.sup:
            logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        
        if not self.sup:
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        else:
            sup_exp_logits = torch.exp(logits[:,:self.num_classes]) * torch.ones(batch_size, self.num_classes).to(device)
            sup_log_prob = logits[:,:self.num_classes] - torch.log(sup_exp_logits.sum(1, keepdim=True) + 1e-12)

            exp_logits = torch.exp(logits[:,self.num_classes:]) * logits_mask
            log_prob = logits[:,self.num_classes:] - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        if self.sup:
            one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
            sup_log_prob_pos = (one_hot_label * sup_log_prob).sum(1) / one_hot_label.sum(1)
            loss = - (self.temperature / self.base_temperature) * (mean_log_prob_pos-mean_log_prob_pos + sup_log_prob_pos)
            loss = loss.mean()

        elif not self.self_sup:
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            sup_logits = (sup_logits - sup_logits).sum()
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean() + sup_logits

        return loss


class ReCoLoss(nn.Module):
    def __init__(self, alpha=1.0, positive_realm_weight=1.0,beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None,K=7, num_classes=1000,relation_dict_file='../ImageNet1K.visual.3_hump.relation.depth_version.json',relation_size=105):
        super(ReCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.supt = supt
        self.num_classes = num_classes
        self.relation_size = relation_size
        self.relation_dict_pro = {}
        self.positive_realm_weight = positive_realm_weight

        with open(relation_dict_file) as f:
                self.relation_dict = json.load(f)

        #Add semantic relation information of each concepts
        for i in self.relation_dict:
            self.relation_dict_pro[i] = list(self.relation_dict[i].values())
            self.relation_dict[i] = list(self.relation_dict[i].keys())

        #For concepts not related to other concepts
        for i in range(self.num_classes):
            if str(i) not in self.relation_dict:
                self.relation_dict[str(i)] = [str(i)]
                self.relation_dict_pro[str(i)] =  1


    def forward(self, features, labels=None,sup_logits=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = ( features.shape[0] - self.K ) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # Get relation logits mask sample certain number of relation pair, default size = 5  
        relation_labels = (-1)*torch.ones(batch_size,self.relation_size,dtype=torch.float).to(device)

        # Get the relation of each instance
        for label_idx,i in enumerate(labels[:batch_size].view(-1).tolist()):
            cur_relation_label = torch.tensor(list(map(lambda x: int(x),self.relation_dict[str(i)]))).to(device)


            cur_relation_label_pro = torch.tensor(list(map(lambda x: float(x),self.relation_dict_pro[str(i)])),dtype=torch.float).to(device)

            if cur_relation_label.size()[0] == 1:
                cur_relation_label_require = cur_relation_label.clone() 
            else:
                idx = torch.cat((torch.tensor([0]), torch.randperm(cur_relation_label.size()[0]-1)+1)).to(device)
                if self.relation_size < cur_relation_label.size()[0]:
                    idx = idx[:self.relation_size]

                cur_relation_label_require = cur_relation_label.clone()
                cur_relation_label_require = cur_relation_label_require[idx]

                cur_relation_label_pro_require = cur_relation_label_pro.clone()
                cur_relation_label_pro_require = cur_relation_label_pro_require[idx]
                #inspired by HCSC:https://github.com/hirl-team/HCSC
                cur_relation_label_pro_require = torch.bernoulli(cur_relation_label_pro_require).to(device)
                cur_relation_label_require = cur_relation_label_require[torch.nonzero(cur_relation_label_pro_require).view(-1)]

            relation_labels[label_idx,:len(cur_relation_label_require)] = cur_relation_label_require.to(device)

        relation_labels_for_query = relation_labels.contiguous().view(-1, 1).expand(-1,len(labels)).clone()

        queue_label = labels.reshape(1,-1).expand(relation_labels_for_query.size()[0],len(labels)).to(device).clone()

        # Get the selected negative samples
        queue_relation_logits_mask = (~(torch.eq(relation_labels_for_query, queue_label).reshape(batch_size,self.relation_size,-1).sum(1).bool())).float().to(device)
        # Add the instances from the same classes (inspired by HCSC:https://github.com/hirl-team/HCSC)
        queue_relation_logits_mask = queue_relation_logits_mask + mask       

        relation_logits_mask = self.gamma * queue_relation_logits_mask

        relation_exp_logits = torch.exp(logits[:,self.num_classes:]) * relation_logits_mask
        relation_log_prob = logits[:,self.num_classes:] - torch.log(relation_exp_logits.sum(1, keepdim=True) + 1e-12)


        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  

       
        # compute mean of log-likelihood over positive for PaCo
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)        

        # compute mean of log-likelihood over positive for ReCo
        mean_log_prob_relation_pos = (mask[:,self.num_classes:] * relation_log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * (mean_log_prob_pos + self.positive_realm_weight * mean_log_prob_relation_pos)

        loss = loss.mean() 
        
        return loss          



def main():
    my_resnet = ReCoLoss(relation_dict_file='./ImageNet1K.visual.3_hump.relation.depth_version.json',relation_size=104,num_classes=30)
    pseudo_input = torch.randn(4*2+7,115217)#.cuda()#109357)
    # pseudo_input = torch.randn(2,3,224,224)
    # pseudo_target = torch.tensor([[2,875],[3,78]],dtype=torch.long)#.cuda()
    pseudo_queue_label = torch.tensor([21,0,23,21,21,0,23,21,1,2,2,23,2,1,0],dtype=torch.long)#.cuda()
    pseudo_softmax = torch.randn(4,30)#.cuda()

    print(my_resnet(pseudo_input,pseudo_queue_label,pseudo_softmax))

if __name__ == '__main__':
    main()

  

