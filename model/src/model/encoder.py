import torch
import torch.nn as nn
import pytorch_lightning as pl

class Encoder(pl.LightningModule):
    """
    Encoder Object
    
    Bidirectional LSTM is used as the encoder.
    Forward and Backwards direction of each layer of Hidden and Cell are combined 
    using a linear layer
    
    """
    def __init__(self, hid_dim, n_layers, n_features, dropout):
        """
        Initialises the encoder object.
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features from the dataset
        :param dropout: dropout ratio for encoder
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_features = n_features
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(n_features, hid_dim, n_layers, dropout = dropout, bidirectional = True,batch_first=True).double()
        
        #linear layers are used to combine the hidden/cell states of the two direction together
        for i in range(n_layers):
            for hc in ['hidden','cell']:
                setattr(self,'{}_linear_{}'.format(hc,i),nn.Linear(hid_dim*2, hid_dim).double())


    
    # def forward(self, x, seq_len):
    #     """
    #     Forward propagation.
    #     :param x: features of dataset at every timestamp [batch_size,sequence_len,feature_dim]
    #     :param seq_len: actual length of each data sequence [batch_size]
    #     :return: hidden state of the last layer in the encoder;
    #              outputs the outputs of last layer at every timestamps
                 
    #     hidden: [batch_size, n_layer, hid_dim]
    #     cell: [batch_size, n_layer, hid_dim]
    #     output [batch_size, sequence_len, n_directions*hid_dim] , note n_direction = 2 for bidirectional-lstm 
    #     """
    #     x = self.dropout(x)
    #     packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_len,batch_first=True)
    #     packed_output, (hidden, cell) = self.rnn(packed_x)
    #     output, _ = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
        
    #     #seperated layer and direction
    #     hidden = hidden.view(self.n_layers, 2, hidden.shape[1], hidden.shape[2]) 
    #     cell = cell.view(self.n_layers, 2, cell.shape[1], cell.shape[2])
        
    #     # for each layer, combine back and forward direction together
    #     all_layer_hidden = []
    #     all_layer_cell = []
    #     for i in range(self.n_layers):
    #         hidden_layer_name = 'hidden_linear_'+str(i)
    #         cell_layer_name = 'cell_linear_{}'.format(str(i))
    #         # stack forward and backward together for each layer
    #         # cell and hidden is not batch first even though argument is passed. [n_direction,batch,hid_dim]
    #         single_layer_hidden = hidden[i]
    #         # permute changes hidden_single_layer to [batch,n_direction,hid_dim]
    #         # view will stack the direction together to [batch,n_direction*hid_dim]
    #         single_layer_hidden = single_layer_hidden.permute(1,0,2).contiguous().view(-1,2*self.hid_dim)
    #         single_layer_cell = cell[i]
    #         single_layer_cell = single_layer_cell.permute(1,0,2).contiguous().view(-1,2*self.hid_dim)
    #         #combine forward and backward together using feedforward layer to [batch,hid_dim] 
    #         all_layer_hidden.append(torch.tanh(getattr(self,hidden_layer_name)(single_layer_hidden)))
    #         all_layer_cell.append(torch.tanh(getattr(self,cell_layer_name)(single_layer_cell)))
    #     #combine list together to form [batch_size,n_layer,hid_dim]
    #     all_layer_hidden = torch.stack(all_layer_hidden,1)
    #     all_layer_cell = torch.stack(all_layer_cell,1)
        
    #     return output, (all_layer_hidden, all_layer_cell)


    def forward(self, x, seq_len):
        """
        NUMBER OF LAYERS HAVE TO BE HARDCODED FOR DEPLOYMENT BECAUSE OF HOW TORCHSCRIPT WORKS.

        Forward propagation.
        :param x: features of dataset at every timestamp [batch_size,sequence_len,feature_dim]
        :param seq_len: actual length of each data sequence [batch_size]
        :return: hidden state of the last layer in the encoder;
                 outputs the outputs of last layer at every timestamps
                 
        hidden: [batch_size, n_layer, hid_dim]
        cell: [batch_size, n_layer, hid_dim]
        output [batch_size, sequence_len, n_directions*hid_dim] , note n_direction = 2 for bidirectional-lstm 
        """
        x = self.dropout(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_len.cpu(),batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
        
        #seperated layer and direction
        hidden = hidden.view(self.n_layers, 2, hidden.shape[1], hidden.shape[2]) 
        cell = cell.view(self.n_layers, 2, cell.shape[1], cell.shape[2])
        
        # for each layer, combine back and forward direction together
        all_layer_hidden = []
        all_layer_cell = []

        #hardcode layer 0

        # stack forward and backward together for each layer
        # cell and hidden is not batch first even though argument is passed. [n_direction,batch,hid_dim]
        single_layer_hidden = hidden[0]
        # permute changes hidden_single_layer to [batch,n_direction,hid_dim]
        # view will stack the direction together to [batch,n_direction*hid_dim]
        single_layer_hidden = single_layer_hidden.permute(1,0,2).contiguous().view(-1,2*self.hid_dim)
        single_layer_cell = cell[0]
        single_layer_cell = single_layer_cell.permute(1,0,2).contiguous().view(-1,2*self.hid_dim)
        #combine forward and backward together using feedforward layer to [batch,hid_dim] 
        all_layer_hidden.append(torch.tanh(self.hidden_linear_0(single_layer_hidden)))
        all_layer_cell.append(torch.tanh(self.cell_linear_0(single_layer_cell)))


        #hardcode layer 1

        # stack forward and backward together for each layer
        # cell and hidden is not batch first even though argument is passed. [n_direction,batch,hid_dim]
        single_layer_hidden = hidden[1]
        # permute changes hidden_single_layer to [batch,n_direction,hid_dim]
        # view will stack the direction together to [batch,n_direction*hid_dim]
        single_layer_hidden = single_layer_hidden.permute(1,0,2).contiguous().view(-1,2*self.hid_dim)
        single_layer_cell = cell[1]
        single_layer_cell = single_layer_cell.permute(1,0,2).contiguous().view(-1,2*self.hid_dim)
        #combine forward and backward together using feedforward layer to [batch,hid_dim] 
        all_layer_hidden.append(torch.tanh(self.hidden_linear_1(single_layer_hidden)))
        all_layer_cell.append(torch.tanh(self.cell_linear_1(single_layer_cell)))



        #combine list together to form [batch_size,n_layer,hid_dim]
        all_layer_hidden_tensor = torch.stack(all_layer_hidden,1)
        all_layer_cell_tensor = torch.stack(all_layer_cell,1)

        return output, (all_layer_hidden_tensor, all_layer_cell_tensor)