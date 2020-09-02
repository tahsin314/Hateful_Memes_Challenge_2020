import logging
logging.basicConfig(level=logging.ERROR)
import transformers
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import torch.nn.functional as F
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
from random import choices
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings('ignore')

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target.astype('int')] = 1.
    return vec

class HMDataset(Dataset):
    def __init__(self, image_ids, texts, tokenizer, max_len=64, labels=None, dim=256, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.transforms = transforms
        self.dim = dim
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.dim, self.dim))
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
        )
        if self.transforms is not None:
            aug = self.transforms(image=image)
            image = aug['image'].reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        else:
            image = image.reshape(self.dim, self.dim, 3).transpose(2, 0, 1)
        if self.labels is not None:
            target = self.labels[idx]
            return image_id, image, text, encoding, onehot(2, target)
        else:
            return image_id, image, text, encoding

    def __len__(self):
        return len(self.image_ids)

    def get_labels(self):
        return list(self.labels)
