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
from torch.optim import AdamW
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
    def __init__(self,config,lora_config):
        super(BrainGPT,self).__init__()
        self.config = config
        self.device = f"cuda:0"
            
        self.encoder = BGRU(encoder_config = config['encoder_config'],
                            keys=sessions,)
        self.encoder = self.encoder.to(self.device)
        self.linear = nn.Linear(config['encoder_config']['hidden_size'],LLM_decoder[config['decoder_config']['name']]['hidden_size']).to(self.device)
        
        self.llm = LLM_decoder[config['decoder_config']['name']]['model'].from_pretrained(LLM_decoder[config['decoder_config']['name']]['path'],
                                                                    torch_dtype=torch.float32,
                                                                    low_cpu_mem_usage=True,
                                                                    device_map = 'auto')
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                inference_mode=False, 
                                r=lora_config['lora_rank'], 
                                lora_alpha=lora_config['lora_alpha'], 
                                lora_dropout=lora_config['lora_dropout'],
                                )
        self.decoder = get_peft_model(self.llm, peft_config)
        self.llm = self.decoder.model
        

    def forward(self,spikepow,labels=None,key=None,pad_token_id = 1):
        if spikepow.dim() == 2:
            spikepow = spikepow.unsqueeze(1)
        
        if labels.dim() == 1:
            spikepow = spikepow.unsqueeze(-1)
        labels = labels.transpose(0,1) 
        encoder_outputs = self.encoder(spikepow,key=key) #L * B * F
        spikepow_embeds = self.linear(encoder_outputs).transpose(0,1)
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
        encoder_outputs = self.encoder(spikepow,key=key)
        spikepow_embeds = self.linear(encoder_outputs).transpose(0,1) # B * L * F
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
    
    def save_decoder_checkpoint(self,path):
        self.decoder.save_pretrained(path)
    
    def load_encoder_checkpoint(self,path):
        self.linear.load_state_dict(torch.load(path + '/linear.pt',map_location = self.device))
        self.encoder.load_from_checkpoint(path)
        print('checkpoint loaded')
    
    def load_decoder_checkpoint(self,path):
        # do not know if it works....
        self.decoder.load_adapter(path,adapter_name = 'BrainLlama_adapter')
        self.llm = self.decoder.model
             

@torch.no_grad()
def evaluate(model,tokenizer,eval_dataloaders):
    device = f"cuda:0"
    model.eval()
    wers = []
    mean_wers_per_day = []
    for key in eval_dataloaders.keys():
        eval_dataloader = eval_dataloaders[key]
        tmp = []
        for src, tgt,src_length,tgt_length in eval_dataloader:
            src = src.to(device) # L*B*F
            tgt = tgt.to(device).transpose(0,1) # B* L
            #print(src.shape)
            #print(tgt.shape)
            outputs = model.generate(src,key = key,tokenizer = tokenizer)
            #print(outputs)
            for i in range(outputs.shape[0]):
                hw = outputs[i,:]
                #if i == 0:
                #    print(hw)
                hw = hw[(hw != tokenizer.eos_token_id) & (hw != tokenizer.pad_token_id)]
                rw = tgt[i,:]
                #if i == 0:
                #    print(rw)
                #rw = rw[rw != -100 and rw != tokenizer.eos_token_id]
                rw = rw[(rw != tokenizer.eos_token_id) & (rw != tokenizer.pad_token_id) &(rw != -100)]
                #if i == 0:
                #    print(hw,rw)
                print(tokenizer.decode(hw.int()))
                print(tokenizer.decode(rw.int()))
                wers.append(calculate_wer(rw,hw))
                tmp.append(wers[-1])
        mean_wers_per_day.append(np.mean(tmp))
    return wers,mean_wers_per_day

if __name__ == '__main__':

    config_path = '/GPFS/data/shengfeng/BrainLlama/config/config.yaml'
    config = get_yaml_data(config_path)
    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['finetune_config']
    lora_config = train_config['lora_config']
    llm_type = model_config['decoder_config']['name']

    print(json.dumps(config))

    print('Start preparing data')
    device = f"cuda:0"
    
    tokenizer =  LLM_decoder[llm_type]['tokenizer'].from_pretrained(LLM_decoder[llm_type]['path'])
    train_dataloaders,val_dataloaders,eval_train_dataloaders,eval_val_dataloaders = prepare_dataloader(data_config['data_path'],sessions,tokenizer,config)

    print('Start loading model')
    model = BrainGPT(model_config,lora_config)
    
    for param in model.parameters():
        param.requires_grad = False
    

    checkpoint_path = train_config['checkpoint_path'] + '/' + ''.join([str(k) + ':' + str(v) + '|' for k,v in model_config['encoder_config'].items()])
    checkpoint_path = checkpoint_path + '/stage2'
    model.load_encoder_checkpoint(checkpoint_path)
    model.load_decoder_checkpoint(checkpoint_path)


    #start_time = timer()
    #train_mean_wers = evaluate(model,tokenizer,eval_train_dataloaders)
    #end_time = timer()
    #print(train_mean_wers)
    #print(f"Train wer: {np.mean(train_mean_wers):.3f} "f"evaluation time = {(end_time - start_time):.3f}s")
    start_time = timer()
    val_mean_wers,val_mean_wers_per_day = evaluate(model,tokenizer,eval_val_dataloaders)
    end_time = timer()
    print(val_mean_wers_per_day)
    print(f"val wer: {np.mean(val_mean_wers):.3f} "f"evaluation time = {(end_time - start_time):.3f}s")
