import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import scipy as sp
import numpy as np
from torch.utils.data import Dataset
import torch
import re
from g2p_en import G2p
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
import math
from timeit import default_timer as timer
from torch.optim.lr_scheduler import PolynomialLR
import torch.nn.functional as F
from tfrecord.torch.dataset import TFRecordDataset
from tqdm import tqdm
from utils import _sessions as sessions
from utils import calculate_wer,get_yaml_data,apply_random_mask,apply_time_reverse,apply_sign_flip,BrainTextDataset,remove_duplicate_and_blank
from model.gru import BGRU
from transformers import get_linear_schedule_with_warmup
import json

torch.manual_seed(0)
np.random.seed(0)

config_path = '/GPFS/data/shengfeng-1/BrainLlama/config/config.yaml'
config = get_yaml_data(config_path)
pretrain_config = config['pretrain_config']
data_config = config['data_config']
model_config = config['model_config']
encoder_config = model_config['encoder_config']

if 'redundant_config' in config.keys():
    if config['redundant_config']['use_redundant']:
        tmp = config['redundant_config']['redundant_features']
        redundant_ids = []
        for i in tmp:
            redundant_ids.append(i)
            redundant_ids.append(i+128)
        useful_ids = [i for i in range(256) if i not in redundant_ids]

DEVICE = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
best_per = 10

print("Start loading data")

data_path = config['data_config']['data_path']

if encoder_config['name'] == 'BGRU':
    train_bpes = []
    test_bpes = []
    with open('/GPFS/data/shengfeng-1/BrainGPT_data/derived/new_transcriptions/train_bpe.txt','r') as f:
        for line in f.readlines():
            train_bpes.append([int(s) for s in line.strip('\n').split(' ')])
    with open('/GPFS/data/shengfeng-1/BrainGPT_data/derived/new_transcriptions/test_bpe.txt','r') as f:
        for line in f.readlines():
            test_bpes.append([int(s) for s in line.strip('\n').split(' ')])

train_data_pairs = {}
val_data_pairs= {}

count_train = 0
count_test = 0
for session in tqdm(sessions):
    train_data_pairs[session] = []
    val_data_pairs[session] = []
    tfrecord_path = data_path + '/' + session
    index_path = None
    tmp_train_dataset = TFRecordDataset(tfrecord_path + '/train/chunk_0.tfrecord', index_path)
    tmp_train_loader = torch.utils.data.DataLoader(tmp_train_dataset, batch_size=1)
    for data in tmp_train_loader:
        input_seq = data['inputFeatures'].reshape(-1,encoder_config['dim_feature'])
        if encoder_config['name'] == 'BGRU':
            #output_seq = data['seqBPEIDs'][0]
            output_seq = torch.Tensor(train_bpes[count_train])
            count_train += 1
        elif encoder_config['name'] == 'BGRU_P':
            output_seq = data['seqClassIDs'][0]
        output_seq = output_seq[output_seq != 0]
        train_data_pairs[session].append([input_seq,output_seq])
    
    tmp_val_dataset = TFRecordDataset(tfrecord_path + '/test/chunk_0.tfrecord', index_path)
    tmp_val_loader = torch.utils.data.DataLoader(tmp_val_dataset, batch_size=1)
    for data in tmp_val_loader:
        input_seq = data['inputFeatures'].reshape(-1,encoder_config['dim_feature'])
        if encoder_config['name'] == 'BGRU':
            #output_seq = data['seqBPEIDs'][0]
            output_seq = torch.Tensor(test_bpes[count_test])
            count_test += 1
        elif encoder_config['name'] == 'BGRU_P':
            output_seq = data['seqClassIDs'][0]
        output_seq = output_seq[output_seq != 0]
        val_data_pairs[session].append([input_seq,output_seq])
        
train_datasets = {}
val_datasets = {}

for key in train_data_pairs.keys():
    train_datasets[key] = BrainTextDataset(train_data_pairs[key])
for key in val_data_pairs.keys():
    val_datasets[key] = BrainTextDataset(val_data_pairs[key])

train_dataloaders = {}
val_dataloaders = {}
eval_train_dataloaders = {}
eval_val_dataloaders = {}

gaussain_kernel_size = int(data_config['sigma']*3)
gaussian_kernel = torch.Tensor([np.exp(-(x ** 2 ) / (2 * data_config['sigma'] ** 2)) for x in range(-gaussain_kernel_size, gaussain_kernel_size + 1)])[None,None,None,:].float()
def collate_fn_stochastic(batch):
    src_batch, tgt_batch,src_length,tgt_length = [], [], [],[]
    for src_sample, tgt_sample in batch:
        src_sample = torch.Tensor(src_sample)
        tgt_sample = torch.Tensor(tgt_sample)
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        src_length.append(src_sample.shape[0]//encoder_config['stride'])
        #print(tgt_sample,tgt_sample.shape)
        tgt_length.append(tgt_sample.shape[0])
    src_batch = pad_sequence(src_batch, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    if data_config['white_noise_sd'] > 0:
        src_batch += np.sqrt(data_config['white_noise_sd'])*torch.randn(src_batch.shape[0],src_batch.shape[1],src_batch.shape[2])
    if data_config['constant_offset_sd'] > 0:
        src_batch += np.sqrt(data_config['constant_offset_sd'])*torch.randn(src_batch.shape[0],1,src_batch.shape[-1])
    src_batch = torch.Tensor(src_batch).float()
    if 'use_random_mask' in encoder_config.keys() and encoder_config['use_random_mask']:
        src_batch = apply_random_mask(src_batch,encoder_config['mask_bandwidth'])
    if 'use_sign_flip' in encoder_config.keys() and encoder_config['use_sign_flip']:
        src_batch = apply_sign_flip(src_batch,encoder_config['sign_flip_proba'])
    if 'use_time_reverse' in encoder_config.keys() and encoder_config['use_time_reverse']:
        src_batch = apply_time_reverse(src_batch,encoder_config['time_reverse_proba'])
    src_batch = F.conv2d(src_batch.unsqueeze(1).transpose(0,-1),gaussian_kernel,padding = 'same').squeeze(1).transpose(0,-1)
    #print(src_batch.shape) # L*B*F
    return src_batch, torch.Tensor(tgt_batch), torch.Tensor(src_length).to(torch.int32),torch.Tensor(tgt_length).to(torch.int32)

def collate_fn(batch):
    src_batch, tgt_batch,src_length,tgt_length = [], [], [],[]
    for src_sample, tgt_sample in batch:
        src_sample = torch.Tensor(src_sample)
        tgt_sample = torch.Tensor(tgt_sample)
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        src_length.append(src_sample.shape[0]//encoder_config['stride'])
        tgt_length.append(tgt_sample.shape[0])
    src_batch = pad_sequence(src_batch, padding_value=0).float()
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    src_batch = F.conv2d(src_batch.unsqueeze(1).transpose(0,-1),gaussian_kernel,padding = 'same').squeeze(1).transpose(0,-1)
    return torch.Tensor(src_batch), torch.Tensor(tgt_batch), torch.Tensor(src_length).to(torch.int32),torch.Tensor(tgt_length).to(torch.int32)

for key in train_datasets.keys():  
    train_dataloaders[key] = DataLoader(train_datasets[key],batch_size = pretrain_config['batch_size'],collate_fn = collate_fn_stochastic,shuffle = True)
for key in val_datasets.keys():
    val_dataloaders[key] = DataLoader(val_datasets[key],batch_size = pretrain_config['batch_size'],collate_fn = collate_fn_stochastic,shuffle = True)

for key in train_datasets.keys():  
    eval_train_dataloaders[key] = DataLoader(train_datasets[key],batch_size = pretrain_config['batch_size'],collate_fn = collate_fn)
for key in val_datasets.keys():
    eval_val_dataloaders[key] = DataLoader(val_datasets[key],batch_size = pretrain_config['batch_size'],collate_fn = collate_fn)

if 'redundant_config' in config.keys():
    if config['redundant_config']['use_redundant']:
        encoder_config['dim_feature'] -= len(redundant_ids)
        
        
print('Start loading model')
keys = list(eval_train_dataloaders.keys())
if encoder_config['name'] == 'BGRU':
    tgt_vocab_size = 100
elif encoder_config['name'] == 'BGRU_P':
    tgt_vocab_size = 40
model = BGRU(encoder_config = encoder_config,
            keys=sessions,
            is_pretraining = True,
            tgt_vocab_size = tgt_vocab_size,
            bidirectional = encoder_config['bidirectional']).float()

model = model.to(DEVICE)
loss_fn = nn.CTCLoss(blank = 0,zero_infinity=True)
optimizer = torch.optim.Adam([{'params':model.GRU.parameters()},
                                {'params':model.generator.parameters()},
                                {'params':model.inputNetworks.parameters(),'weight_decay':0}],
                            lr=pretrain_config['lr_encoder'], betas=(0.9, 0.999), eps=pretrain_config['lr_encoder'],weight_decay = pretrain_config['l2_coef'])
num_steps = 0
for key in train_dataloaders.keys():
    train_dataloader = train_dataloaders[key]
    num_steps += len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = pretrain_config['warmup_steps'],num_training_steps = num_steps*pretrain_config['num_epochs'])

def train_epoch(model, optimizer,train_dataloaders):
    model.train()
    losses = 0
    for key in train_dataloaders.keys():
        train_dataloader = train_dataloaders[key]
        for src, tgt,src_length,tgt_length in train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_length = src_length.to(DEVICE)
            tgt_length = tgt_length.to(DEVICE)
            if 'redundant_config' in config.keys():
                if config['redundant_config']['use_redundant']:
                    src =  src[:,:,useful_ids]
            logits = torch.log(model(src,key))
            optimizer.zero_grad()

            loss = loss_fn(logits, tgt.transpose(0,1),src_length,tgt_length)
            reg = None
            for n,p in model.named_parameters():
                if key in n:
                    if reg is None:
                        reg = p.norm(2)
                    else:
                        reg = reg + p.norm(2)
                        
            loss += reg * pretrain_config['l2_coef']
            loss.backward()
            if pretrain_config['clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_config['clip_norm'])
            optimizer.step()
            scheduler.step()
            losses += loss.item()
    return losses/num_steps

@torch.no_grad()
def evaluate(model,val_dataloaders):
    model.eval()
    pers = []
    for key in val_dataloaders.keys():
        val_dataloader = val_dataloaders[key]
        for src, tgt,src_length,tgt_length in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_length = src_length.to(DEVICE)
            tgt_length = tgt_length.to(DEVICE)
            if 'redundant_config' in config.keys():
                if config['redundant_config']['use_redundant']:
                    src =  src[:,:,useful_ids]
            logits = torch.log(model(src,key))
            tgt_pred = logits.squeeze().argmax(dim = -1)
            if tgt_pred.dim() == 1:
                tgt_pred = tgt_pred.unsqueeze(-1)
            for i in range(src.shape[1]):
                rl = tgt_length[i]
                hl = src_length[i]
                rw = tgt[:rl,i].tolist()
                hw = remove_duplicate_and_blank(tgt_pred[:hl,i].tolist())
                pers.append(calculate_wer(rw,hw))
    return np.mean(pers)

count = 0
stop_flag = False
checkpoint_path = pretrain_config['checkpoint_path'] +'/'+ ''.join([str(k) + ':' + str(v) + '|' for k,v in model_config['encoder_config'].items()])+ '/pretrain'
if pretrain_config['save_checkpoint']:
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open(checkpoint_path + '/config.json','wt') as f:
        json.dump(config,f,indent=4)
        
if pretrain_config['is_monitoring']:
    loss_list = []
    val_per_list = []
print("Start training")
EPOCHS_PER_EVAL = 1
for epoch in range(1, pretrain_config['num_epochs']+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer,train_dataloaders)
    end_time = timer()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
    if (epoch) % EPOCHS_PER_EVAL == 0:
        start_time = timer()
        #train_per = evaluate(model,eval_train_dataloaders)
        val_per = evaluate(model,eval_val_dataloaders)
        end_time = timer()
        print(f"Epoch: {epoch},Val per: {val_per:.3f} "f"evaluation time = {(end_time - start_time):.3f}s")
        if best_per > val_per:
            best_per = val_per
            model.save_checkpoint(checkpoint_path)
            print('checkpoint saved')
            count = 0
        else:
            count += 1
            if pretrain_config['early_stopping']:
                if count > pretrain_config['patience']:
                    stop_flag = True
        if pretrain_config['is_monitoring']:
            loss_list.append(train_loss)
            val_per_list.append(val_per)
            result_config = {
                'loss':loss_list,
                'val_per':val_per_list,
                'best_per':best_per,
            }
            with open(checkpoint_path + '/result.json','wt') as f:
                json.dump(result_config,f,indent=4)    
        if stop_flag:
            print('Early stopping')
            break