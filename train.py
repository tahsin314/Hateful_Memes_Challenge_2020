import logging
logging.basicConfig(level=logging.ERROR)
from config import *
from utils import *
import gc
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch import optim
from transformers import BertTokenizer
from HMDataset import HMDataset
from model.img.resne_t import Resne_t
from model.hybrid import Hybrid 

if mixed_precision:
  scaler = torch.cuda.amp.GradScaler() 

np.random.seed(SEED)
tokenizer = BertTokenizer.from_pretrained(nlp_model_name)

train_df = pd.read_json(f'{data_dir}/train.jsonl', lines=True)
val_df = pd.read_json(f'{data_dir}/dev.jsonl', lines=True)

train_df['img'] = train_df['img'].map(lambda x: f"{data_dir}{x}")
val_df['img'] = val_df['img'].map(lambda x: f"{data_dir}/{x}")
history = pd.DataFrame()
train_ds = HMDataset(train_df.img.values, train_df.text.values, tokenizer, max_len, train_df.label.values, dim=img_dim, transforms=train_aug)
val_ds = HMDataset(val_df.img.values, val_df.text.values, tokenizer, max_len, val_df.label.values, dim=img_dim, transforms=val_aug)

train_loader = DataLoader(train_ds,batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

# model = Resne_t(img_model_name).to(device)
model = Hybrid(img_model_name, nlp_model_name).to(device)
criterion = nn.BCEWithLogitsLoss()


def train_val(epoch, dataloader, optimizer, rate = 1.00, train=True, mode='train'):
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
    for idx, (_, img,text,encoding, labels) in enumerate(dataloader):
        with torch.set_grad_enabled(train):
            img = img.to(device)
            input_ids = encoding['input_ids'].view(-1, max_len).to(device)
            attention_mask = encoding['attention_mask'].view(-1, max_len).to(device)
            labels = labels.to(device)
            epoch_samples += len(img)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(mixed_precision):
            # outputs_img = model(img.float())
            # loss_img = criterion(outputs_img, labels)
            # Calculate loss for encoded text here
            # loss_text = 0
            outputs = model(img, input_ids, attention_mask)
            # loss = loss_img + loss_text
            loss = criterion(outputs, labels)
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

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
        # Replace outputs_img with outputs
        pred.extend(torch.softmax(outputs, 1)[:,1].detach().cpu().numpy())
        lab.extend(torch.argmax(labels, 1).cpu().numpy())
        if train:
            msg = f"Epoch: {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s"
        else:
            msg = f'Epoch {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(running_loss/epoch_samples):.4f} Time: {elapsed}s ETA: {eta} s'
        print(msg, end= '\r')
    history.loc[epoch, f'{mode}_loss'] = running_loss/epoch_samples
    history.loc[epoch, f'{mode}_time'] = elapsed
    history.loc[epoch, 'rate'] = rate  
    if mode=='val':
        auc = roc_auc_score(lab, pred)
        lr_reduce_scheduler.step(running_loss)
        msg = f'{mode} Loss: {running_loss/epoch_samples:.4f} \n {mode} Auc: {auc:.4f}'
        print(msg)
        history.loc[epoch, f'{mode}_loss'] = running_loss/epoch_samples
        history.loc[epoch, f'{mode}_auc'] = auc
        history.to_csv(f'{history_dir}/history_{model_name}_{img_dim}.csv', index=False)
        return running_loss/epoch_samples, auc


plist = [ 
        {'params': model.img_model.parameters(),  'lr': learning_rate/100},
        {'params': model.nlp_model.parameters(),  'lr': learning_rate/1000},
        {'params': model.out.parameters(),  'lr': learning_rate},
    ]
optimizer = optim.Adam(plist, lr=learning_rate)
lr_reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

def main():
    prev_epoch_num = 0
    best_valid_loss = np.inf
    best_valid_auc = 0.0

    if load_model:
        tmp = torch.load(os.path.join(model_dir, model_name+'_tasn_loss.pth'))
        model.load_state_dict(tmp['model'])
        optimizer.load_state_dict(tmp['optim'])
        lr_reduce_scheduler.load_state_dict(tmp['scheduler'])
        # cyclic_scheduler.load_state_dict(tmp['cyclic_scheduler'])
        scaler.load_state_dict(tmp['scaler'])
        prev_epoch_num = tmp['epoch']
        best_valid_loss = tmp['best_loss']
        best_valid_loss, best_valid_auc = train_val(prev_epoch_num+1, valid_loader, optimizer=optimizer, rate=1, train=False, mode='val')
        del tmp
        print('Model Loaded!')
  
    for epoch in range(prev_epoch_num, n_epochs):
        torch.cuda.empty_cache()
        print(gc.collect())
        # rate = 1
        # if epoch < 20:
        # rate = 1
        # elif epoch>=20 and rate>0.65:
        # rate = np.exp(-(epoch-20)/40)
        # else:
        # rate = 0.65

        train_val(epoch, train_loader, optimizer=optimizer, rate=rate, train=True, mode='train')
        valid_loss, valid_auc = train_val(epoch, valid_loader, optimizer=optimizer, rate=1.00, train=False, mode='val')
        print("#"*20)
        print(f"Epoch {epoch} Report:")
        print(f"Validation Loss: {valid_loss :.4f} Validation AUC: {valid_auc :.4f}")
        best_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler':lr_reduce_scheduler.state_dict(), 
        
        # 'cyclic_scheduler':cyclic_scheduler.state_dict(), 
            'scaler': scaler.state_dict(),
        'best_loss':valid_loss, 'best_auc':valid_auc, 'epoch':epoch}
        best_valid_loss, best_valid_auc = save_model(valid_loss, valid_auc, best_valid_loss, best_valid_auc, best_state, os.path.join(model_dir, model_name))
        print("#"*20)
   
if __name__== '__main__':
  main()