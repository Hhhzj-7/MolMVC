import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool, global_add_pool
from method import  GNN,Encoder_MultipleLayers,SchNet 
from torch_geometric.utils import  degree
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add
from method.model_helper import Encoder_1d, Embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
SAVE = 'save/model_pre20.pth'


class transformer_1d(nn.Sequential):
    def __init__(self):
        super(transformer_1d, self).__init__()
        input_dim_drug = 2587
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,
                         transformer_dropout_rate)

        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    def forward(self, emb, mask):
        e = emb.long().to(device)
        e_mask = mask.long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers, _ = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers

class projection_head(nn.Module):
    def __init__(self):
        super(projection_head, self).__init__()
        self.line1 = nn.Linear(128, 128)
        self.line2 = nn.Linear(128, 64)

        self.relu = nn.ReLU()
    
    def forward(self, emb):
        out1 = self.line1(emb)
        out = self.relu(out1)
        out = self.line2(out)

        return out, out1
    
def floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    adj_mat_copy = adjacency_matrix.astype(float, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    M = adj_mat_copy

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if M[i][j] > M[i][k] + M[k][j]:
                    M[i][j] = M[i][k] + M[k][j]
                    # path[i][j] = k

    return M


def get_spatil_pos(batch_node, edge_index_s, batch=None):
    if type(batch_node) != list:
        batch_node = batch_node.tolist()
    N = 0
    row, col = [], []
    adj = torch.zeros([1])
    spe = []
    N_last = 0
    for x in range(batch_node[len(batch_node)-1] + 1):
        N = batch_node.count(x)
        edge_index = batch[x].edge_index
        N_last = N_last + N

        adj = torch.zeros([N, N], dtype=torch.bool)
        row, col = edge_index
        adj[row, col] = True
        shortest_path_result = floyd_warshall(adj.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        if N < 50:
            spatial_pos = torch.nn.functional.pad(spatial_pos, pad=(0,50-N,0,50-N), value=0)     
        else:
            spatial_pos= spatial_pos[:50,:50]
        spe.append(spatial_pos)
    spe = torch.stack(spe).to(device)
    return spe

def get_3d_pos(pos, batch_batch):
    batch_node = batch_batch.tolist()
    edge_index = radius_graph(pos, r=10.0, batch=batch_batch)
    row, col = edge_index
    dist = (pos[row] - pos[col]).norm(dim=-1)

    atom_dist_add = scatter_add(dim=0, index=edge_index[1], src=dist)

    atom_flag = 0
    dist_m = []

    

    for x in range(batch_node[len(batch_node)-1] + 1):
        pos_t = pos[atom_flag:(atom_flag+batch_node.count(x))]
        if pos_t.size()[0] == 0:
            dist_m.append(torch.zeros((50, 50)).to(device))
            atom_flag += batch_node.count(x)
            continue
        edge_index_temp = radius_graph(pos_t, r=10.0)

        row_t, col_t = edge_index_temp
        dist_t = (pos_t[row_t] - pos_t[col_t]).norm(dim=-1)
        dist_zero = torch.zeros((50, 50)).to(device)
        
        dist_t = dist_zero.index_put([row_t, col_t], dist_t)
        dist_m.append(dist_t)
        atom_flag += batch_node.count(x)

    dist_m = torch.stack(dist_m).to(device)
    return atom_dist_add, dist_m


class CL_model(nn.Module):
    def __init__(self, device, is_match, only_2d):
        super(CL_model, self).__init__()

        self.max = 0

        self.device = device
        self.is_match = is_match
        self.only_2d = only_2d
        
        # Multiple GPUs need to remove ‘modules’
        model_dict = torch.load(SAVE,map_location=device)
        new_state_dict = {}
        for k,v in model_dict['cl_model'].items():
            new_state_dict[k[7:]] = v # 去掉module
        
        self.model_3d = SchNet().to(device)
        self.model_2d = GNN().to(device)
        self.model_2d_high = Transformer_E().to(device)
        self.model_3d_high = Transformer_E().to(device)
        self.model_1d = transformer_1d().to(device)
        self.projection_2d_low = projection_head().to(device)
        self.projection_2d_high = projection_head().to(device)
        
        self.cl_model = CL_model_2d().to(device)
        
        # load multi
        self.cl_model.load_state_dict(new_state_dict)
        
        self.line1 = nn.Linear(128, 64) # 256 384
        self.line2 = nn.Linear(64, 32)

        self.pre = nn.Linear(32,1)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, batch, get_emb=False):
        emb_2d_low, emb_2d_high, emb_1d_low, emb_1d_high = self.cl_model(batch)  

        # low and high
        out_2d = (emb_2d_low + emb_2d_high) / 2
        out_1d = (emb_1d_low + emb_1d_high) / 2

        
        out = torch.concat((out_1d, out_2d), dim=1)

        out = self.dropout(self.relu(self.line1(out_2d)))
        out = self.dropout(self.relu(self.line2(out)))
        out = self.pre(out)
        if get_emb:
            return out, out_1d, out_2d
        else: 
            return out

class Transformer_E(nn.Sequential):
    def __init__(self):
        super(Transformer_E, self).__init__()
        transformer_emb_size_drug = 128
        transformer_n_layer_drug = 4
        transformer_intermediate_size_drug = 256
        transformer_num_attention_heads_drug = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1


        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)

    def forward(self, h_node, mask, if_2d,in_degree_2d=None, out_degree_2d=None, dist_3d=None, dist_m=None, spe=None):
        encoded_layers, _, all = self.encoder(h_node.float(), mask.float(),if_2d,in_degree_2d=in_degree_2d, out_degree_2d=out_degree_2d,dist_3d=dist_3d, dist_m=dist_m, spe=spe)
        return encoded_layers, all
    
    
class CL_model_2d(nn.Module):
    def __init__(self):
        super(CL_model_2d, self).__init__()
        self.model_2d = GNN()
        self.model_3d= SchNet()
        self.model_T_2d = Transformer_E()
        self.model_T_3d = Transformer_E()
        self.model_1d = transformer_1d()
        self.projection_2d_low = projection_head()
        self.projection_2d_high = projection_head()
        self.projection_3d_low = projection_head()
        self.projection_3d_high = projection_head()
        self.projection_1d_low = projection_head()
        self.projection_1d_high = projection_head()
    def forward(self, batch):
        # 1d
        smile_emb = torch.from_numpy(np.asarray(batch.smiles))
        smile_mask = torch.from_numpy(np.asarray(batch.mask))
        emb_1d = self.model_1d(smile_emb, smile_mask)
        emb_1d_low = emb_1d[3]
        emb_1d_high = emb_1d[7]
        
        # suit for transfomer encoder
        out_2d, out_2d_res, _ = self.model_2d(batch.x, batch.edge_index, batch.edge_attr)
        emb_2d_low = global_mean_pool(out_2d, batch.batch)
        out_2d = out_2d + out_2d_res
        batch_node = batch.batch.tolist()
        h_node_two = []
        mask = []
        mask_out = []
        in_degree_2d_final = []
        out_degree_2d_final = []
        in_degree_2d = degree(batch.edge_index[0], num_nodes=len(batch.x)).int()
        out_degree_2d = degree(batch.edge_index[1], num_nodes=len(batch.x)).int()
        spe = get_spatil_pos(batch.batch, batch.edge_index, batch=batch)

        flag = 0
        # some mole is too long
        for x in range(batch_node[len(batch_node)-1] + 1):
            if batch_node.count(x) < 50:
                mask.append([] + batch_node.count(x) * [0] + (50 - batch_node.count(x)) * [-10000])
                mask_out.append([] + batch_node.count(x) * [1] + (50 - batch_node.count(x)) * [0])
            else:
                mask.append([] + 50 * [0])
                mask_out.append([] + 50 * [1])

            oral_node_2d = out_2d[flag:flag + batch_node.count(x)]
            in_degree_2d_oral = in_degree_2d[flag:flag + batch_node.count(x)]
            out_degree_2d_oral = out_degree_2d[flag:flag + batch_node.count(x)]
            flag += batch_node.count(x)
            if batch_node.count(x) < 50:
                temp_2d = torch.full([(50-oral_node_2d.size()[0]), 128], 0).to(device)
                temp_in_degree_2d = torch.full([(50-oral_node_2d.size()[0])], 0).to(device)
                temp_out_degree_2d = torch.full([(50-oral_node_2d.size()[0])], 0).to(device)
                in_degree_2d_oral = torch.cat((in_degree_2d_oral, temp_in_degree_2d),0)
                out_degree_2d_oral = torch.cat((out_degree_2d_oral, temp_out_degree_2d),0)
                final_node_2d = torch.cat((oral_node_2d, temp_2d),0)                
            else:
                final_node_2d = oral_node_2d[:][:][:50]
                in_degree_2d_oral = in_degree_2d_oral[:][:][:50]
                out_degree_2d_oral = out_degree_2d_oral[:][:][:50]

                
            h_node_two.append(final_node_2d)
            in_degree_2d_final.append(in_degree_2d_oral)
            out_degree_2d_final.append(out_degree_2d_oral)
            
        h_node_two = torch.stack(h_node_two).to(device)
        mask_2d = torch.tensor(mask, dtype=torch.float)
        mask_2d =  mask_2d.to(device).unsqueeze(1).unsqueeze(2)
        mask_2d_out = torch.tensor(mask_out, dtype=torch.float).to(device)
        mask_1d_out = smile_mask.to(device)
        in_degree_2d_final = torch.stack(in_degree_2d_final).to(device)
        out_degree_2d_final = torch.stack(out_degree_2d_final).to(device)


        # 2d transformer emb
        emb_2d_high, _ = self.model_T_2d(h_node_two, mask_2d, True, in_degree_2d_final, out_degree_2d_final, spe=spe)
        emb_2d_high_out2 = torch.div(torch.sum(torch.mul(emb_2d_high ,mask_2d_out.unsqueeze(2)),dim=1), torch.sum(mask_2d_out, dim=1).unsqueeze(1))
        # 1d transormer emb
        emb_1d_low = torch.div(torch.sum(torch.mul(emb_1d_low ,mask_1d_out.unsqueeze(2)),dim=1), torch.sum(mask_1d_out, dim=1).unsqueeze(1))
        emb_1d_high = torch.div(torch.sum(torch.mul(emb_1d_high ,mask_1d_out.unsqueeze(2)),dim=1), torch.sum(mask_1d_out, dim=1).unsqueeze(1))

        emb_2d_high = emb_2d_high_out2
        

        return emb_2d_low, emb_2d_high, emb_1d_low, emb_1d_high
