import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
class Attention(pl.LightningModule):
    '''
    Attention Object
    This Implments the ADDITIVE attention e = W3tanh(W1q + W2k) , c = sum(softmax(e)v)
    # q = prevois hidden, k = encoding outputs , v = encoding outouts
    '''
    def __init__(self, hid_dim):
        """
        Initialises the attention object.
        :param hid_dim: hidden dimensions in each layer
        """
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim).double()
        self.v = nn.Linear(hid_dim, 1, bias = False).double()
        
    def forward(self, last_hidden, encoder_output, mask):
        """
        Forward propagation.
        :param last_hidden: hidden state of last layer from previous timestamp (tensor) [batch,hid_dim]
        :param encoder_output: used to measure similiarty in states [batch,sequence_len,hid_dim*2]
        :return: normalized probabilities for each timestamp - softmax (tensor)
        
        returns [batch_size,sequence_len]
        """
        #last hidden layer [batch_size,hid_dim] 
        src_len = encoder_output.shape[1]
        # expand to [batch_size, sequence_len, hid_dim]
        last_hidden = last_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # after cat the size is [batch_size,sequence_len,3*hid_dim]
        # energy is [batch_size,sequence_len,3*hid_dim]
        energy = torch.tanh(self.attn(torch.cat((last_hidden, encoder_output), dim = 2))) 
        # attention is [batch_size,sequence_len,1], after squeeze its [batch_size,sequence_len]
        attention = self.v(energy).squeeze(2) 
        attention = attention.masked_fill(~mask, -1e10)
        return F.softmax(attention, dim=1)