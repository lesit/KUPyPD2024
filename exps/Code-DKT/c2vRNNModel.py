# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

class c2vRNNModel(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, layer_dim, output_dim, emb_size,
                 node_count, path_count, max_code_len):
        super(c2vRNNModel, self).__init__()
        
        self.max_code_len = max_code_len

        self.embed_nodes = nn.Embedding(node_count+2, emb_size) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, emb_size) # adding unk and end

        self.code_feature_indices = (input_dim, input_dim + 3*max_code_len)

        self.embed_dropout = nn.Dropout(0.2)

        code_embed_concat_size = emb_size*3 # start node, path node, end node
        
        self.path_transformation_layer = nn.Linear(input_dim+code_embed_concat_size, input_dim+code_embed_concat_size)
        self.attention_layer = nn.Linear(input_dim+code_embed_concat_size,1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(2*input_dim+code_embed_concat_size,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device


    def forward(self, x):  # shape of input: [batch_size, length, questions*2 + c2vnodes]
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        
        rnn_first_part = x[:, :, :self.input_dim] # (b,l,2q)
        rnn_attention_part = torch.stack([rnn_first_part]*self.max_code_len,dim=-2) # (b,l,c,2q)

        c2v_input = x[:, :, self.code_feature_indices[0]:self.code_feature_indices[1]].reshape(x.size(0), x.size(1), self.max_code_len, 3).long() # (b,l,c,3)

        starting_node_index = c2v_input[:,:,:,0] # (b,l,c,1)
        ending_node_index = c2v_input[:,:,:,2] # (b,l,c,1)
        path_index = c2v_input[:,:,:,1] # (b,l,c,1)

        starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,c,1) -> (b,l,c,ne)
        ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,c,1) -> (b,l,c,ne)
        path_embed = self.embed_paths(path_index) # (b,l,c,1) -> (b,l,c,pe)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed, rnn_attention_part), dim=3) # (b,l,c,2ne+pe+q)

        full_embed = self.embed_dropout(full_embed) 
        
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed))
        context_weights = self.attention_layer(full_embed_transformed) # (b,l,c,1)
        attention_weights = self.attention_softmax(context_weights) # (b,l,c,1)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2) 
        rnn_input = torch.cat((rnn_first_part,code_vectors), dim=2)
        
#         print(rnn_input.shape)
        out, hn = self.rnn(rnn_input)  # shape of out: [batch_size, length, hidden_size]
#         out, hn = self.rnn(x, h0)  # shape of out: [batch_size, length, hidden_size]
#         res = self.sig(self.fc(self.dropout(out)))  # shape of res: [batch_size, length, question]
        res = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]
        return res
