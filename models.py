import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
import torch.nn.init as init
from torch.nn import functional as F

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000 ,device1='cuda"1',**kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.device=device1
        self.base_network = backbones.get_backbone(base_net,self.device)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.feature_dim, bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.feature_dim
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_head = 4
        for i in range(self.num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())

    def forward(self, source, target):
        s_logits, s_p, s_attn = self.base_network(source)  # 32*512
        t_logits, t_p, t_attn = self.base_network(target)
        # s_feat_4, source = self.base_network(source)  # 32*2048*7*7, 32*2048
        # t_feat_4, target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(s_logits)  #32*256
            target = self.bottleneck_layer(t_logits)
        # classification
        source_clf = self.classifier_layer(source)  #32*6
        target_clf = self.classifier_layer(target)
        #clf_loss = self.criterion(source_clf, source_label)
        
        N=s_p.size()[0]
        s_p = s_p.transpose(1, 2)
        s_p = s_p.view(N,1024,7,7)
        heads_s = []
        
        for i in range(self.num_head):
            heads_s.append(getattr(self,"cat_head%d" %i)(s_p))
            # heads_s.append(getattr(self, "cat_head%d" % i)(source.unsqueeze(0).unsqueeze(0)))
        
        heads_s = torch.stack(heads_s).permute([1,0,2])
        if heads_s.size(1)>1:
            heads_s = F.log_softmax(heads_s,dim=1)
        
        # transfer
        kwargs = {}
        
        # transfer_loss = self.adapt_loss(s_feat_4, t_feat_4, **kwargs)
        transfer_loss = self.adapt_loss(source,target, **kwargs)
        return source, source_clf, target, target_clf, transfer_loss, heads_s
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features,_,_ = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass
        
class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca
    
class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, x):  # 32*2048*7*7
        y = self.conv1x1(x)   #32*256*7*7
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        
        return out

class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(1024, 32),# (512,32)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1024), #(32,512)
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        
        return out