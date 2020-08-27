from copy import deepcopy
from config import *
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from typing import Optional        
import timm
from pprint import pprint

class Resne_t(nn.Module):

    def __init__(self, model_name='resnet34'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.in_features = 512
        self.backbone.fc = nn.Linear(self.in_features, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x, meta_data=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.global_pool(x)
        x = self.backbone.fc(x)
        x = self.out(x)
        return x