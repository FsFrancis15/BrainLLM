import torch.nn as nn
import torch

class BGRU(nn.Module):
    def __init__(self,
                 encoder_config,
                 keys,
                 is_pretraining = False,
                 tgt_vocab_size = 100,
                 bidirectional = False,
                 short_token = False,
                ):
        super(BGRU, self).__init__()

        num_layers = encoder_config['num_layers']
        hidden_size = encoder_config['hidden_size']
        stride = encoder_config['stride']
        dim_feature = encoder_config['dim_feature']
        dropout_gru = encoder_config['dropout_gru']
        dropout_inp = encoder_config['dropout_inp']
        kernel_size = encoder_config['kernel_size']

        self.GRU = torch.nn.GRU(dim_feature*kernel_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=dropout_gru, bidirectional=bidirectional, device=None, dtype=None)
        self.stride = stride
        self.dim_feature = dim_feature
        self.kernel_size = kernel_size
        self.dropout_inp = dropout_inp
        self.extract_patches = nn.Unfold((self.kernel_size,1), dilation=1, padding=((self.kernel_size - 4)//2,0), stride=(self.stride,1))
        self.keys = keys
        self._buildInputNetworks()
        self.generator = None
        self.short_token = short_token
        '''if is_pretraining:
            tgt_vocab_size = 4096
            self.generator = nn.Sequential(nn.Linear(hidden_size, tgt_vocab_size),
                                            #nn.Softmax(dim = -1),
            )'''
        
        if is_pretraining:
            if bidirectional:
                self.generator = nn.Sequential(nn.Linear(hidden_size*2, tgt_vocab_size + 1),
                                                nn.Softmax(dim = -1),
                                                )
            else:
                self.generator = nn.Sequential(nn.Linear(hidden_size, tgt_vocab_size + 1),
                                                nn.Softmax(dim = -1),
                                                )
        for n,p in self.named_parameters():
            if p.dim() > 1:
                if 'GRU' in n:
                    nn.init.orthogonal_(p)
                elif 'inputNetwork' in n:
                    nn.init.eye_(p)
                else:
                    nn.init.xavier_uniform_(p)

    def _buildInputNetworks(self):
        if self.keys is not None:
            inputNetworks = {}
            for key in self.keys:
                inputNetworks[key] = nn.Sequential(nn.Linear(self.dim_feature,self.dim_feature),
                                                    nn.Dropout(self.dropout_inp),
                                                    nn.Softsign())
            self.inputNetworks = nn.ParameterDict(inputNetworks)
        else:
            self.inputNetwork = nn.Sequential(nn.Linear(self.dim_feature,self.dim_feature),
                                                    nn.Dropout(self.dropout_inp),
                                                    nn.Softsign())

    def load_from_checkpoint(self,checkpoint_dir,load_generator = False):
        self.GRU.load_state_dict(torch.load(checkpoint_dir+'/gru.pt',map_location = next(self.parameters()).device))
        self.inputNetworks = torch.load(checkpoint_dir+'/imp.th',map_location = next(self.parameters()).device)
        if load_generator:
            self.generator.load_state_dict(torch.load(checkpoint_dir+'/generator.pt',map_location = next(self.parameters()).device))
        
    def save_checkpoint(self,checkpoint_dir):
        torch.save(self.generator.state_dict(),checkpoint_dir+'/generator.pt')
        torch.save(self.GRU.state_dict(),checkpoint_dir+'/gru.pt')
        torch.save(self.inputNetworks,checkpoint_dir+'/imp.th')
        
        
    def forward(self,
                src,
                key = None):
        if self.keys is not None:
            src_emb = self.inputNetworks[key](src)
        else:
            src_emb = self.inputNetwork(src)
        src_emb = self.extract_patches(src_emb.permute(1,2,0).unsqueeze(-1))
        src_emb = src_emb.permute(2,0,1)
        outs, hidden_states = self.GRU(src_emb)
        if self.short_token:
            if self.generator is not None:
                hidden_states = self.generator(hidden_states)
            return hidden_states
        else:
            if self.generator is not None:
                outs = self.generator(outs)
            return outs