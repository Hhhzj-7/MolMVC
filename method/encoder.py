import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import nn
import torch.nn.functional as F
import copy
import math




class distance_NN(nn.Module):
    def __init__(self, hidden_size=50, num_layers=2):
        super(distance_NN, self).__init__()

        non_lin_fun = torch.nn.GELU
        layers = [torch.nn.Linear(1, hidden_size), non_lin_fun()]
        for kk in range(num_layers):
            layers += [torch.nn.Linear(hidden_size, hidden_size),
                       non_lin_fun()]
        layers += [torch.nn.Linear(hidden_size, 4)]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, D_mat):
        # return self.seq(D_mat.unsqueeze(-1)).squeeze(-1)
        return self.seq(D_mat.unsqueeze(-1))


# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

# word embedding and position encoding
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
# self attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob,num_in_degree=50,num_out_degree=50):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads    # multi-heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.spatial_pos_encoder = 512

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # self.query2 = nn.Linear(hidden_size, self.all_head_size)
        # self.key2 = nn.Linear(hidden_size, self.all_head_size)
        # self.value2 = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        # position encoding for 2d
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_size, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_size, padding_idx=0)
        # position encoding for 3d
        self.net_3d_encoder = torch.nn.Sequential(*[nn.Linear(1, hidden_size,
                                                   bias=True), torch.nn.GELU(), nn.Linear(hidden_size, 1,
                                                                                          bias=True),
                                         torch.nn.GELU()])
        self.embed_3d = nn.Linear(1, hidden_size,
                               bias=True)

        self.att_dist = distance_NN(50,2)


        
        self.spatial_pos_encoder = nn.Embedding(self.spatial_pos_encoder, num_attention_heads)
    
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, if_2d, in_degree, out_degree, dist_3d, dist_m, spe):
        
        
        
        if if_2d == True:
            pos_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            attention_bias = self.spatial_pos_encoder(spe).permute(0, 3, 1, 2)
        else:            
            dist_3d = torch.unsqueeze(dist_3d, -1)
            pos_feature = self.net_3d_encoder(dist_3d)
            pos_feature = self.embed_3d(pos_feature)
            
            attention_bias = self.att_dist(dist_m).permute(0, 3, 1, 2)

        
        #     print(1)
        #     print(pos_feature[0][0])
        # print(2)
        # print(hidden_states[0][0])
        hidden_states = hidden_states + pos_feature

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # if if_2d == True:
        #     spe_encoding = self.spatial_pos_encoder(spe).permute(0, 3, 1, 2)

        #     # print(0)
        #     # print(attention_scores[0][0][0])
        #     # print(1)
        #     # print(spe_encoding[0][0][0])
        #     attention_scores = attention_scores + spe_encoding
        #     # print(spe.size())

        # else:
        #     dist_attn = self.att_dist(dist_m).permute(0, 3, 1, 2)
            
            
        #     # print(dist_attn.size())
        #     # dist_attn = dist_m.unsqueeze(1).repeat(1,
        #     #                                     self.num_attention_heads,
        #     #                                     1,
        #     #                                     1)
        #     # print(attention_scores.size(), dist_attn.size())
        #     # print(2)
        #     # print(attention_scores[0][0][0])
        #     # print(3)
        #     # print(dist_attn[0][0][0])
            
        #     attention_scores = attention_scores + dist_attn

        attention_scores = attention_scores + attention_bias

        attention_scores = attention_scores + attention_mask
        

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print(attention_probs)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
            
        return context_layer, attention_probs
    
# output of self-attention
class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
# attention layer  
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask, if_2d, in_degree_2d, out_degree_2d, dist_3d, dist_m, spe):
        self_output, attention_scores = self.self(input_tensor, attention_mask, if_2d, in_degree_2d, out_degree_2d, dist_3d, dist_m, spe)  # 这里实际上出来了2个，后面是不是每个需要训练的层都要两份分开
        # if if_2d:
        #     input_tensor = torch.cat((input_tensor[0].unsqueeze(0), input_tensor[1].unsqueeze(0)),0)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores    
    
# after attention    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # self.act = swish

    def forward(self, hidden_states, if_2d):
        if if_2d:
            hidden_states = self.dense(hidden_states)
            hidden_states = F.relu(hidden_states)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = F.gelu(hidden_states)
        return hidden_states
# output
class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Transformer encoder
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, if_2d, in_degree_2d, out_degree_2d, dist_3d, dist_m, spe):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask, if_2d, in_degree_2d, out_degree_2d, dist_3d, dist_m, spe)
        intermediate_output = self.intermediate(attention_output, if_2d)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_scores    

# multi-heads transformer encoder
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, if_2d, in_degree_2d=None, out_degree_2d=None, output_all_encoded_layers=True, dist_3d=None, dist_m=None, spe=None):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask, if_2d, in_degree_2d, out_degree_2d, dist_3d, dist_m, spe)
            all_encoder_layers.append(hidden_states)
        return hidden_states, attention_scores, all_encoder_layers
    