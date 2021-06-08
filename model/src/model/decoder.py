import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple
class Decoder(pl.LightningModule):
    """
    Decoder Object
    
    Note that the forward method only steps forward in time ONCE. 
    It expects an input of [batch]
    """
    def __init__(self, output_dim, hid_dim, n_layers, dropout, attention, mode3_encoder, callsign_encoder):
        """
        Initialises the decoder object.
        :param output_dim: number of classes to predict
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param dropout: dropout ratio for decoder
        :param attention: attention object to used (initialized in seq2seq)
        """
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.mode3_encoder = mode3_encoder
        self.callsign_encoder = callsign_encoder
        self.embedding = nn.Embedding(output_dim + 2 , hid_dim).double() # we have two special tokens
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + hid_dim + hid_dim + hid_dim, output_dim).double()
        self.rnn = nn.LSTM(hid_dim * 2 + hid_dim, hid_dim, n_layers, dropout = dropout,batch_first=True).double()

    def forward(self, decoder_input, mode3_input, callsign_input, hidden_cell: Tuple[Tensor,Tensor], encoder_output, mask):
        """
        Forward propagation.
        :param input: label of dataset at each timestamp (tensor) [batch_size]
        :param hidden_cell: hidden state from previous timestamp (tensor) ([batch_size,n_layer,hid_dim],[batch_size,n_layer,hid_dim]) 
        :param encoder_output: used to measure similiarty in states in attention [batch_size,sequence_len,hid_dim]
        :param mask: mask to filter out the paddings in attention object [batch_size,sequence_len]
        :return: normalized output probabilities for each timestamp - softmax (tensor) [batch_size,sequence_len,num_outputs]
        """
        hidden = hidden_cell[0]
        last_hidden = hidden[:,-1,:]
        a = self.attention(last_hidden, encoder_output, mask)
        a = a.unsqueeze(1) #(batch,1, time_normalized)
        #this is the weighted sum of encoder outputs

        weighted = torch.bmm(a, encoder_output) #(batch,1,hidden_dim) 
        
        #input is [batch] 
        decoder_input = decoder_input.unsqueeze(1) #[batch,1]
        embedded = self.embedding(decoder_input) #[batch,1,hid_dim]
        embedded = self.dropout(embedded)

        rnn_input = torch.cat((embedded, weighted), dim = 2) 
        # we can add information on the callsign here 


        #hidden_cell is (batch,layer,hid_dim) but pytorch requires (layer,batch,hid_dim)
        
        hidden_cell = (hidden_cell[0].permute(1,0,2).contiguous() ,hidden_cell[1].permute(1,0,2).contiguous())
        
        output, (hidden, cell) = self.rnn(rnn_input, hidden_cell) #hidden equals
        #reshape for batch_first consistency
        hidden_cell = (hidden.permute(1,0,2),cell.permute(1,0,2))
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        # we can also add information on the callsign here 
        mode3_embedding = self.mode3_encoder(mode3_input) 
        callsign_embedding = self.callsign_encoder(callsign_input) 
        prediction = self.fc_out(torch.cat((output,weighted,embedded,mode3_embedding,callsign_embedding), dim = 1))
        return F.softmax(prediction,dim=1), hidden_cell