import numpy as np
import torch.nn as nn
from torchvision import models
#swim
from timm.models.swin_transformer import SwinTransformer
from timm import create_model
from timm.models.registry import register_model


from clipvit.clip_part import *
import torch
import torchvision
from transformers import ViTImageProcessor, ViTModel
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name,device):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "vit1" in name.lower():
        return VitBackbone(device)
    elif "swimvit" in name.lower():
        return Swim_base(device)

class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)
        x = self.avgpool(x1)
        x = x.view(x.size(0), -1)
        return x1, x
    
    def output_num(self):
        return self._feature_dim

# class VitBackbone(nn.Module):
#     def __init__(self):
#         super(VitBackbone, self).__init__()
#         # 加载预训练的 ViT 模型和特征提取器
#         # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
#         # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.feature_extractor = ViTImageProcessor.from_pretrained(
#             '/data/guoshuo/vit_pre_weight/vit-base-patch16-224-in21k')
#         self.model = ViTModel.from_pretrained('/data/guoshuo/vit_pre_weight/vit-base-patch16-224-in21k')
#         self.vit=self.model.encoder
#
#     def forward(self,x):
#         # patch_size = 16  # 图块大小
#         # x= x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
#         # x = x.contiguous().view(-1, 3, patch_size, patch_size)  # (num_patches, 3, patch_size, patch_size)
#
#         print(x[1])
#         x = self.feature_extractor(x)
#         x=self.vit(x)
#         x = x.view(x.size(0), -1)
#         print(x)
#         return x
#
#     def output_num(self):
#         return self.vit.config.hidden_size

class VitBackbone(nn.Module):
    def __init__(self,device):
        super(VitBackbone, self).__init__()
        clip_model = load_clip_to_cpu(device)
        clip_model.float()
        self.image_encoder = clip_model.encode_image


    def forward(self,x):

        x = self.image_encoder(x)

        return x

    def output_num(self):

        return 512
  
class Swin(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_features(self, x, patch=False):
        if not patch:
            x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        patch = x
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x, patch, None
      
@register_model
def ds_swin_base_patch4_window7_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model = Swin(**model_kwargs)
    if pretrained:
        pre = create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        model.load_state_dict(pre.state_dict())
        
    return model

class Swim_base(nn.Module):
    def __init__(self,device):
        super(Swim_base, self).__init__()
        model_path = '/DATA/hejinbo/qiyuhan/vit_FER/pretrained_models/swin_base_patch4_window7_224_22kto1k.pth'
        self.backbone = create_model('ds_swin_base_patch4_window7_224', pretrained=True)
        self.feature_dim = self.backbone.num_features
        self.num_patch = self.backbone.patch_embed.num_patches
        self.num_classes = 6

    def forward(self,x):

        token = self.backbone.patch_embed(x)
        logits, p, attn = self.backbone.forward_features(token, patch=True)

        return logits, p, attn