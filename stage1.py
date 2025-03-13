import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
from model.gru import BGRU
import argparse
from utils import prepare_dataloader,calculate_wer,get_yaml_data
from utils import _sessions as sessions
from utils import _LLM_decoder as LLM_decoder
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW,Adam
import torch.nn as nn
import torch
from timeit import default_timer as timer
import datetime
from torchmetrics.text import WordErrorRate
import numpy as np
from transformers import StoppingCriteria,StoppingCriteriaList
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType,LoraConfig
import logging
from copy import deepcopy
logging.basicConfig(level='ERROR')
torch.manual_seed(0)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

class BrainGPT(nn.Module):
    def __init__(self,config):
        super(BrainGPT,self).__init__()
        self.config = config
        self.device = f"cuda:0"
        
        self.encoder = BGRU(encoder_config = config['encoder_config'],
                            keys=sessions,
                            bidirectional = config['encoder_config']['bidirectional'])
        self.encoder = self.encoder.to(self.device)
        if config['encoder_config']['bidirectional']:
            self.linear = nn.Linear(config['encoder_config']['hidden_size']*2,LLM_decoder[config['decoder_config']['name']]['hidden_size']).to(self.device)
        else:
            self.linear = nn.Linear(config['encoder_config']['hidden_size'],LLM_decoder[config['decoder_config']['name']]['hidden_size']).to(self.device)

        for param in self.linear.parameters():
            if param.dim() >1:
                torch.nn.init.eye_(param)
        
        self.llm = LLM_decoder[config['decoder_config']['name']]['model'].from_pretrained(LLM_decoder[config['decoder_config']['name']]['path'],
                                                                    torch_dtype=torch.float32,
                                                                    low_cpu_mem_usage=True,
                                                                    device_map = 'auto')
        #self.fusion_rate = config['encoder_config']['fusion_rate']
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self,spikepow,labels=None,key=None,pad_token_id = 1):
        if spikepow.dim() == 2:
            spikepow = spikepow.unsqueeze(1)
        
        if labels.dim() == 1:
            spikepow = spikepow.unsqueeze(-1)
        labels = labels.transpose(0,1) 
        spikepow_embeds = self.encoder(spikepow,key=key).transpose(0,1) #B * L * F
        #if self.fusion_rate > 1:
        #    spikepow_embeds = torch.nn.functional.avg_pool1d(spikepow_embeds.transpose(1,2),kernel_size=self.fusion_rate,stride=self.fusion_rate).transpose(1,2)
        spikepow_embeds = self.linear(spikepow_embeds)
        spikepow_ids = (-100*torch.ones((spikepow_embeds.shape[0],spikepow_embeds.shape[1]))).to(self.device)

        labels = labels.to(self.device)
        labels_ = labels.clone()
        labels_[labels_==-100] = pad_token_id
        #tgt_embeds = self.gpt2.transformer.wte(labels_.int())
        if 'gpt2' in self.config['decoder_config']['name']:
            tgt_embeds = self.llm.transformer.wte(labels_.int())
        elif 'opt' in self.config['decoder_config']['name']:
            tgt_embeds = self.llm.model.decoder.embed_tokens(labels_.int())
        elif 'llama' in self.config['decoder_config']['name']:
            tgt_embeds = self.llm.model.embed_tokens(labels_.int())
        inputs_embeds = torch.cat((spikepow_embeds,tgt_embeds),dim = 1) # B * L * F
        #print(inputs_embeds.shape)
        labels = torch.cat((spikepow_ids,labels),dim = 1) # B * L
        #print(labels.shape) 
        #print(inputs_embeds.shape)
        #print(labels.shape)
        #outputs = self.llm(inputs_embeds = inputs_embeds.long(),labels = labels.long(),return_dict = True)
        outputs = self.llm(inputs_embeds = inputs_embeds,labels = labels.to(torch.int64),return_dict = True)
        #outputs = self.llm(inputs_embeds = inputs_embeds,labels = labels,return_dict = True)
        return outputs

    def generate(self,spikepow,key,tokenizer):
        if spikepow.dim() == 2:
            spikepow = spikepow.unsqueeze(1)
        
        stopping_criteria = None
        if 'gpt' in self.config['decoder_config']['name']:
            stop_words_ids = [tokenizer.eos_token_id]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])
        spikepow_embeds = self.encoder(spikepow,key=key).transpose(0,1)
        #if self.fusion_rate > 1:
        #    spikepow_embeds = torch.nn.functional.avg_pool1d(spikepow_embeds.transpose(1,2),kernel_size=self.fusion_rate,stride=self.fusion_rate).transpose(1,2)

        spikepow_embeds = self.linear(spikepow_embeds) # B * L' * F
        batch_size,seq_len,feature_len = spikepow_embeds.shape[0],spikepow_embeds.shape[1],spikepow_embeds.shape[2]

        start_token_id = torch.ones((batch_size,1)).to(self.device)*tokenizer.bos_token_id 
        if 'gpt2' in self.config['decoder_config']['name']:
            start_token_embed = self.llm.transformer.wte(start_token_id.int())
        elif 'opt' in self.config['decoder_config']['name']:
            start_token_embed = self.llm.model.decoder.embed_tokens(start_token_id.int())
        elif 'llama' in self.config['decoder_config']['name']:
            start_token_embed = self.llm.model.embed_tokens(start_token_id.int())
        spikepow_embeds = torch.cat((spikepow_embeds,start_token_embed),dim = 1)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        outputs = self.llm.generate(inputs_embeds = spikepow_embeds,
                                    max_new_tokens = 20,
                                    pad_token_id=pad_token_id,
                                    stopping_criteria = stopping_criteria,
                                    )
        #print(spikepow_embeds.shape)
        #print(outputs.shape)
        return outputs

    def save_encoder_checkpoint(self,path):
        torch.save(self.linear.state_dict(),path+'/linear.pt')
        torch.save(self.encoder.GRU.state_dict(),path+'/gru.pt')
        torch.save(self.encoder.inputNetworks,path+'/imp.th')
        print('checkpoint saved')
    
    def load_encoder_checkpoint(self,path):
        self.linear.load_state_dict(torch.load(path + '/linear.pt',map_location = self.device))
        self.encoder.load_from_checkpoint(path)
        print('checkpoint loaded')
             


def train_epoch(model,optimizer,scheduler,train_dataloaders,num_steps):
    device = f"cuda:0"
    #device = 'cpu'
    model.train()
    losses = 0
    for key in train_dataloaders.keys():
        train_dataloader = train_dataloaders[key]
        for src, tgt,src_length,tgt_length in train_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            outputs = model(src,labels = tgt,key = key)
            loss = torch.mean(outputs['loss'])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses += loss.item()
    return losses/num_steps

@torch.no_grad()
def evaluate(model,tokenizer,eval_dataloaders):
    device = f"cuda:0"
    model.eval()
    wers = []
    for key in eval_dataloaders.keys():
        eval_dataloader = eval_dataloaders[key]
        for src, tgt,src_length,tgt_length in eval_dataloader:
            src = src.to(device) # L*B*F
            tgt = tgt.to(device).transpose(0,1) # B* L
            outputs = model.generate(src,key = key,tokenizer = tokenizer)
            for i in range(outputs.shape[0]):
                 
                #hw = outputs[i,:]
                #hw = hw[(hw != tokenizer.eos_token_id) & (hw != tokenizer.pad_token_id)]
                #rw = tgt[i,:]
                #rw = rw[(rw != tokenizer.eos_token_id) & (rw != tokenizer.pad_token_id) &(rw != -100)]
                
                hw = outputs[i,:]
                hw = hw[(hw != tokenizer.eos_token_id) & (hw != tokenizer.pad_token_id)]
                rw = tgt[i,:]
                rw = rw[(rw != tokenizer.eos_token_id) & (rw != tokenizer.pad_token_id) &(rw != -100)]
                hw  = tokenizer.decode(hw.int())
                rw = tokenizer.decode(rw.int())
                
                hw = hw.lower().replace('.','').replace(',','').replace('?','').replace('-',' ').replace(';','').replace('!','').replace(':','').replace('"','')
                rw = rw.lower().replace('.','').replace(',','').replace('?','').replace('-',' ').replace(';','').replace('!','').replace(':','').replace('"','')
                
                hw = hw.split(' ')
                rw = rw.split(' ')
                wers.append(calculate_wer(rw,hw))
                
    return np.mean(wers)

if __name__ == '__main__':

    config_path = '/GPFS/data/shengfeng-1/BrainLlama/config/config.yaml'
    config = get_yaml_data(config_path)
    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']
    llm_type = model_config['decoder_config']['name']

    print(json.dumps(config))

    print('Start preparing data')
    device = f"cuda:0"
    
    tokenizer =  LLM_decoder[llm_type]['tokenizer'].from_pretrained(LLM_decoder[llm_type]['path'])
    train_dataloaders,val_dataloaders,eval_train_dataloaders,eval_val_dataloaders = prepare_dataloader(data_config['data_path'],sessions,tokenizer,config)

    print('Start loading model')
    model = BrainGPT(model_config)
    
    param_list = []
    param_list.append({'params':model.linear.parameters(),'lr':train_config['lr_linear']})
    param_list.append({'params':model.encoder.parameters(),'lr':train_config['lr_encoder']})

    optimizer = Adam(param_list,eps = train_config['eps'])

    num_steps = 0
    
    for key in train_dataloaders.keys():
        train_dataloader = train_dataloaders[key]
        num_steps += len(train_dataloader)
    #scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = train_config['warmup_steps'],num_training_steps = num_steps*train_config['num_epochs'])
    scheduler = CosineAnnealingLR(optimizer,T_max = num_steps*train_config['num_epochs'])
    print('Start training')
    epochs_per_eval = 1
    best_val_wer = 10

    count = 0
    stop_flag = False
    checkpoint_path = train_config['checkpoint_path'] + '/' + ''.join([str(k) + ':' + str(v) + '|' for k,v in model_config['encoder_config'].items()])
    model.encoder.load_from_checkpoint(config['encoder_checkpoint'])
    ## for CL only
    ## model.linear = torch.nn.Sequential(model.linear)
    ## model.linear.load_state_dict(torch.load(checkpoint_path + '/pretrain/generator.pt',map_location = next(model.parameters()).device))
    
    checkpoint_path = checkpoint_path + '/stage1'
    if train_config['continue']:
         model.load_encoder_checkpoint(checkpoint_path)
    if train_config['save_checkpoint']:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open(checkpoint_path + '/config.json','wt') as f:
            json.dump(config,f,indent=4)

    if train_config['is_monitoring']:
        val_wer_list = []
        loss_list = []
        if train_config['continue']:
            with open(checkpoint_path + '/result.json') as f:
                result_config = json.load(f)
                val_wer_list = result_config['val_wer']
                loss_list = result_config['loss']
                best_val_wer = result_config['best_wer']

    for epoch in range(1,train_config['num_epochs']+1):
        start_time = timer()
        train_loss = train_epoch(model,optimizer,scheduler,train_dataloaders,num_steps)
        end_time = timer()
        if train_config['is_monitoring']:
            loss_list.append(train_loss)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))
        if (epoch) % epochs_per_eval == 0:
            #start_time = timer()
            #train_mean_wer = evaluate(model,tokenizer,eval_train_dataloaders,args)
            #end_time = timer()
            #print(f"Epoch: {epoch},Train wer: {train_mean_wer:.3f} "f"evaluation time = {(end_time - start_time):.3f}s")
            start_time = timer()
            val_mean_wer = evaluate(model,tokenizer,eval_val_dataloaders)
            end_time = timer()
            if train_config['is_monitoring']:
                val_wer_list.append(val_mean_wer)
            if val_mean_wer < best_val_wer:
                best_val_wer = val_mean_wer
                count = 0
                if train_config['save_checkpoint']:
                    model.save_encoder_checkpoint(checkpoint_path)
            else:
                count += 1
                if train_config['early_stopping']:
                    if count > train_config['patience']:
                        stop_flag = True
            print(f"Epoch: {epoch},Best val wer: {best_val_wer:.3f},Current val wer: {val_mean_wer:.3f} "f"evaluation time = {(end_time - start_time):.3f}s")
        if stop_flag:
            print('Early stopping')
            break
        if train_config['is_monitoring']:
            result_config = {
                'loss':loss_list,
                'val_wer':val_wer_list,
                'best_wer':best_val_wer,
            }
            with open(checkpoint_path + '/result.json','wt') as f:
                json.dump(result_config,f,indent=4)    



