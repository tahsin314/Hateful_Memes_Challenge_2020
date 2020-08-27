from config import *
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch import optim
from HMDataset import HMDataset
from model.img.resne_t import Resne_t

if mixed_precision:
  scaler = torch.cuda.amp.GradScaler() 

np.random.seed(SEED)

train_df = pd.read_json(f'{data_dir}/train.jsonl', lines=True)
val_df = pd.read_json(f'{data_dir}/dev.jsonl', lines=True)

train_ds = HMDataset(train_df.img.values, train_df.text.values, train_df.label.values, dim=img_dim, transforms=train_aug)
val_ds = HMDataset(val_df.img.values, val_df.text.values, val_df.label.values, dim=img_dim, transforms=val_aug)

train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

model = Resne_t(model_name).to(device)
criterion = nn.BCEWithLogitsLoss()

def train_val(epoch, dataloader, optimizer, train=True, mode='train'):
    t1 = time.time()
    running_loss = 0
    epoch_samples = 0
    pred = []
    lab = []
    if train:
        model.train()
        print("Initiating train phase ...")
    else:
        model.eval()
        print("Initiating val phase ...")
    for idx, (_, img,text,labels) in enumerate(dataloader):
        with torch.set_grad_enabled(train):
            img = img.to(device)
            #  load encoded text here
            labels = labels.to(device)
            epoch_samples += len(img)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(mixed_precision):
            outputs_img = model(img.float())
            loss_img = criterion(outputs_img, labels)
            # Calculate loss for encoded text here
            loss_text = 0
            loss = loss_img + loss_text
            running_loss += loss.item()

            if train:
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer) 
                    scaler.update() 
                    optimizer.zero_grad()
                    # cyclic_scheduler.step()
            else:
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

plist = [ 
        {'params': model.backbone.parameters(),  'lr': learning_rate/100},
        {'params': model.out.parameters(),  'lr': learning_rate},
    ]
optimizer = optim.Adam(plist, lr=learning_rate)