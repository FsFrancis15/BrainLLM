from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel,GPT2Tokenizer,OPTForCausalLM,AutoTokenizer,AutoModelForCausalLM
import yaml
from tqdm import tqdm

_sessions = [
  't12.2022.04.28',
  't12.2022.05.05',
  't12.2022.05.17',
  't12.2022.05.19',
  't12.2022.05.24',
  't12.2022.05.26',
  't12.2022.06.02',
  't12.2022.06.07',
  't12.2022.06.14',
  't12.2022.06.16',
  't12.2022.06.21',
  't12.2022.06.23', # non-vocal
  't12.2022.06.28',
  't12.2022.07.05',
  't12.2022.07.14',
  't12.2022.07.21',
  't12.2022.07.27',
  't12.2022.07.29', # vocal
  't12.2022.08.02',
  't12.2022.08.11',
  't12.2022.08.13',
  't12.2022.08.18', # non-vocal
  't12.2022.08.23', # non-vocal
  't12.2022.08.25', # non-vocal
]


_LLM_decoder = {
    'gpt2':
        {'path':'/GPFS/data/shengfeng-1/models/GPT-2',
         'model':GPT2LMHeadModel,
         'tokenizer':GPT2Tokenizer,
         'hidden_size':768,
        },
    'gpt2-large':
        {'path':'/GPFS/data/shengfeng-1/models/GPT2-Large',
         'model':GPT2LMHeadModel,
         'tokenizer':GPT2Tokenizer,
         'hidden_size':1280,
        },
    'gpt2-xl':
        {'path':'/GPFS/data/shengfeng-1/models/GPT2-Xlarge',
         'model':GPT2LMHeadModel,
         'tokenizer':GPT2Tokenizer,
         'hidden_size':1600,
        },
    'opt-350m':
        {'path':'/GPFS/data/shengfeng-1/models/OPT-350m',
         'model':OPTForCausalLM,
         'tokenizer':AutoTokenizer,
         'hidden_size':512,
        },
    'opt-1.3b':
        {'path':'/GPFS/data/shengfeng-1/models/OPT-1.3b',
         'model':OPTForCausalLM,
         'tokenizer':AutoTokenizer,
         'hidden_size':2048,  
        },
        
    'opt-2.7b':
        {'path':'/GPFS/data/shengfeng-1/models/OPT-2.7b',
         'model':OPTForCausalLM,
         'tokenizer':AutoTokenizer,
         'hidden_size':2560,  
        },
        
    'opt-6.7b':
        {'path':'/GPFS/data/shengfeng-1/models/OPT-6.7b',
         'model':OPTForCausalLM,
         'tokenizer':AutoTokenizer,
         'hidden_size':4096, 
        },
    
    'llama2-7b':
        {'path':'/GPFS/data/shengfeng-1/models/Llama2-7b',
         'model':AutoModelForCausalLM,
         'tokenizer':AutoTokenizer,
         'hidden_size':4096,  
        }
}

class BrainTextDataset(Dataset):
    def __init__(self,data_pair):
        self.data_pair = data_pair

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self,idx):
        return self.data_pair[idx]
    
class collate_fn_(object):
    def __init__(self, stride = 4, sigma = 2,is_stochastic = True,white_noise_sd = 1,constant_offset_sd = 0.2,mask_bandwidth = 40):
        self.stride = stride
        if is_stochastic:
            # use data augmentation
            self.white_noise_sd = white_noise_sd
            self.constant_offset_sd = constant_offset_sd
            self.mask_bandwidth = mask_bandwidth
        else:
            self.white_noise_sd = -1
            self.constant_offset_sd = -1
            self.mask_bandwidth = -1
        gaussain_kernel_size = int(sigma*3)
        self.gaussian_kernel = torch.Tensor([np.exp(-(x ** 2 ) / (2 * sigma ** 2)) for x in range(-gaussain_kernel_size, gaussain_kernel_size + 1)])[None,None,None,:].float()

    def __call__(self,batch):
        src_batch, tgt_batch,src_length,tgt_length = [], [], [],[]
        for src_sample, tgt_sample in batch:
            src_sample = torch.Tensor(src_sample)
            tgt_sample = torch.Tensor(tgt_sample)
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
            src_length.append(src_sample.shape[0]//self.stride)
            tgt_length.append(tgt_sample.shape[0])
            assert src_length[-1] >= 1
            assert tgt_length[-1] >= 1
        src_batch = pad_sequence(src_batch, padding_value=0)
        tgt_batch = pad_sequence(tgt_batch, padding_value=-100)
        if self.white_noise_sd > 0:
            src_batch += np.sqrt(self.white_noise_sd)*torch.randn(src_batch.shape[0],src_batch.shape[1],src_batch.shape[2])
        if self.constant_offset_sd > 0:
            src_batch += np.sqrt(self.constant_offset_sd)*torch.randn(src_batch.shape[0],1,src_batch.shape[-1])
        src_batch = torch.Tensor(src_batch).float()
        if self.mask_bandwidth > 0:
            src_batch = apply_random_mask(src_batch,self.mask_bandwidth)
        ## for debug use
        src_batch = F.conv2d(src_batch.unsqueeze(1).transpose(0,-1),self.gaussian_kernel,padding = 'same').squeeze(1).transpose(0,-1)
        return src_batch, torch.Tensor(tgt_batch), torch.Tensor(src_length).to(torch.int32),torch.Tensor(tgt_length).to(torch.int32)

def prepare_dataloader(data_path,sessions,tokenizer,config):
    train_data_pairs = {}
    val_data_pairs = {}

    dim_feature = config['model_config']['encoder_config']['dim_feature']
    batch_size = config['train_config']['batch_size']
    stride = config['model_config']['encoder_config']['stride']
    LLM = config['model_config']['decoder_config']['name']
    sigma = config['data_config']['sigma']
    white_noise_sd = config['data_config']['white_noise_sd']
    constant_offset_sd = config['data_config']['constant_offset_sd']
    mask_bandwidth = config['data_config']['mask_bandwidth']
    
    simple_flag = False
    if 'simple' in config['model_config']['encoder_config']['name']:
        print('Use simple sentences')
        simple_flag = True
        train_sentences = []
        test_sentences = []
        with open('/GPFS/data/shengfeng/BrainGPT_data/derived/new_transcriptions/train.txt','r') as f:
            for line in f.readlines():
                train_sentences.append(line.strip('\n'))
        with open('/GPFS/data/shengfeng/BrainGPT_data/derived/new_transcriptions/test.txt','r') as f:
            for line in f.readlines():
                test_sentences.append(line.strip('\n'))
                
    if simple_flag:
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
            spikepow = data['inputFeatures'].reshape(-1,dim_feature)
            if simple_flag:
                sentence = train_sentences[count_train]
                count_train += 1
            else:
                sentence = data['transcription_text'][0].decode()
            if 'gpt' in LLM:
                text = [tokenizer.bos_token_id] + tokenizer.encode(sentence) + [tokenizer.eos_token_id]
            else:
                text = tokenizer.encode(sentence) + [tokenizer.eos_token_id]
            train_data_pairs[session].append([spikepow,text])
        
        tmp_val_dataset = TFRecordDataset(tfrecord_path + '/test/chunk_0.tfrecord', index_path)
        tmp_val_loader = torch.utils.data.DataLoader(tmp_val_dataset, batch_size=1)
        for data in tmp_val_loader:
            spikepow = data['inputFeatures'].reshape(-1,dim_feature)
            if simple_flag:
                sentence = test_sentences[count_test]
                count_test += 1
            else:
                sentence = data['transcription_text'][0].decode()
            if 'gpt' in LLM:
                text = [tokenizer.bos_token_id] + tokenizer.encode(sentence) + [tokenizer.eos_token_id]
            else:
                #for opt
                text = tokenizer.encode(sentence) + [tokenizer.eos_token_id]
            val_data_pairs[session].append([spikepow,text])

    train_datasets = {}
    val_datasets = {}

    for key in train_data_pairs.keys():
        train_datasets[key] = BrainTextDataset(train_data_pairs[key])
    for key in val_data_pairs.keys():
        val_datasets[key] = BrainTextDataset(val_data_pairs[key])

    collate_fn_stochastic = collate_fn_(stride = stride, 
                                        sigma = sigma,
                                        is_stochastic = True,
                                        white_noise_sd = white_noise_sd,
                                        constant_offset_sd = 
                                        constant_offset_sd,
                                        mask_bandwidth = mask_bandwidth)
    collate_fn = collate_fn_(stride = stride,is_stochastic = False)
    train_dataloaders = {}
    val_dataloaders = {}
    eval_train_dataloaders = {}
    eval_val_dataloaders = {}
    for key in train_datasets.keys():  
        train_dataloaders[key] = DataLoader(train_datasets[key],batch_size = batch_size,collate_fn = collate_fn_stochastic,shuffle = True)
    for key in val_datasets.keys():
        val_dataloaders[key] = DataLoader(val_datasets[key],batch_size = batch_size,collate_fn = collate_fn_stochastic,shuffle = True)

    # fix the batch size of evaluation
    for key in train_datasets.keys():  
        eval_train_dataloaders[key] = DataLoader(train_datasets[key],batch_size = batch_size,collate_fn = collate_fn)
    for key in val_datasets.keys():
        eval_val_dataloaders[key] = DataLoader(val_datasets[key],batch_size = batch_size,collate_fn = collate_fn)

    return train_dataloaders,val_dataloaders,eval_train_dataloaders,eval_val_dataloaders

def calculate_wer(ref_words, hyp_words):
    # Split the reference and hypothesis sentences into words
    # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
    # The extra row and column are for the case when one of the strings is empty
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # The number of operations for an empty hypothesis to become the reference
    # is just the number of words in the reference (i.e., deleting all words)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    # The number of operations for an empty reference to become the hypothesis
    # is just the number of words in the hypothesis (i.e., inserting all words)
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    # Iterate over the words in the reference and hypothesis
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            # If the current words are the same, no operation is needed
            # So we just take the previous minimum number of operations
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                # If the words are different, we consider three operations:
                # substitution, insertion, and deletion
                # And we take the minimum of these three possibilities
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    # The minimum number of operations to transform the hypothesis into the reference
    # is in the bottom-right cell of the matrix
    # We divide this by the number of words in the reference to get the WER
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer
    
def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.safe_load(file_data)
    return data

def apply_time_reverse(src_batch,p):
    if np.random.uniform(0,1)<p:
        return torch.from_numpy(src_batch.numpy()[::-1,:,:].copy())
    else:
        return src_batch
    
def apply_random_mask(src_batch,bandwidth):
    mask = torch.ones(src_batch.shape)
    length = src_batch.shape[0]
    start = np.random.randint(length-bandwidth)
    mask[start:start+bandwidth,:,:] = 0
    return src_batch*mask

def apply_sign_flip(src_batch,p):
    if np.random.uniform(0,1)<p:
        return -src_batch
    else:
        return src_batch
    
def remove_duplicate_and_blank(lyst):
    w = lyst[0]
    i = 1
    while(i<len(lyst)-1):
        if lyst[i] == w:
            lyst = lyst[:i] + lyst[i+1:]
        else:
            w = lyst[i]
            i += 1
    lyst = np.array(lyst)
    lyst = lyst[lyst!=0]
    return lyst.tolist()
