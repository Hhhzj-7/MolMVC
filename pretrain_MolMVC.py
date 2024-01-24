#coding=utf-8
import os
import torch
import torch_geometric
from torch import nn
from method import GNN, Encoder_MultipleLayers, SchNet
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
from process_dataset import PCQM4Mv2Dataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_add
from torch.utils.data.sampler import SubsetRandomSampler
from AMCLoss import AMCLoss
from method.model_helper import Encoder_1d, Embeddings
import copy  
torch.set_printoptions(precision=None, threshold=10000, edgeitems=None, linewidth=None, profile=None)
torch.set_printoptions(threshold=np.inf)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device_ids = [0, 1]
device = "cuda" if torch.cuda.is_available() else "cpu"
output_model_dir = 'save/'
EPOCH = 20
BATCH_SIZE = 400 * len(device_ids)
MASK_RATIO = 0.3
molecule_readout_func = global_mean_pool

# Initialize AMCLoss
last_loss_low = torch.zeros(50,15) # 15 pairs
last_loss_high = torch.zeros(50,15)
AMC_loss_low = AMCLoss(last_loss_low).to(device)
AMC_loss_high = AMCLoss(last_loss_high).to(device)



# Transformer encoder for ESPF
class transformer_1d(nn.Sequential):
    def __init__(self):
        super(transformer_1d, self).__init__()
        input_dim_drug = 2586 + 1 # last for mask
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
        encoded_layers, attention_scores = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers, attention_scores

# Generate 1D attention-guided mask with pior knowledge
def get_mask_indices_1d(attention_scores, s, mask):
    s_c = copy.deepcopy(s)
    m_c = copy.deepcopy(mask)
    attention_sorted, rank_indices = torch.sort(attention_scores, dim=1, descending=True)
    for i in range(len(s_c)):
        size = m_c[i].tolist().count(1)
        temp = rank_indices[i][:size] # get true node; from high to low
        # attention_guided record
        top_temp = temp[:int(size * MASK_RATIO)].tolist()
        # mask
        s_c[i][top_temp] = 2586
    return s_c



# Transformer encoder for 2D and 3D graph transformer 
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
        encoded_layers, attention_scores,_ = self.encoder(h_node.float(), mask.float(),if_2d ,in_degree_2d=in_degree_2d, out_degree_2d=out_degree_2d,dist_3d=dist_3d, dist_m=dist_m, spe=spe)
        return encoded_layers, attention_scores


# SPD for 2D
def get_spatil_pos(batch_node, edge_index_s, batch=None):
    batch_node = batch_node.tolist()
    N = 0
    row, col = [], []
    adj = torch.zeros([1])
    spe = []
    for x in range(batch_node[len(batch_node)-1] + 1):
        N = batch_node.count(x)
        if batch == None:
            edge_index = edge_index_s[x]
        else:
            edge_index = batch[x].edge_index
        adj = torch.zeros([N, N], dtype=torch.bool)
        row, col = edge_index
        adj[row, col] = True
        shortest_path_result = floyd_warshall(adj.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        spatial_pos = torch.nn.functional.pad(spatial_pos, pad=(0,50-N,0,50-N))
        spe.append(spatial_pos)
    spe = torch.stack(spe).to(device)
    return spe



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

    return M


# Distance for 3D
def get_3d_pos(pos, batch_batch):
    batch_node = batch_batch.tolist()
    edge_index = radius_graph(pos, r=10.0, batch=batch_batch)
    row, col = edge_index
    dist = (pos[row] - pos[col]).norm(dim=-1)

    atom_dist_add = scatter_add(dim=0, index=edge_index[1], src=dist)

    atom_flag = 0
    dist_m = []

    # First, calculate the number of each molecular node, and then calculate the number of edges contained in each node
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


# processing output of 2D and 3D transformer encoder
def get_transformer_emb(batch, emb_2d, emb_3d, in_degree_2d, out_degree_2d, dist_3d, dist_m,spe, model_T_2d, model_T_3d):
    # suit for transfomer encoder
    batch_node = batch.tolist()
    h_node_two = []
    h_node_three = []
    in_degree_2d_final = []
    out_degree_2d_final = []
    dist_3d_final = []
    mask = []
    mask_out = []
    flag = 0
    for x in range(batch_node[len(batch_node)-1] + 1):
        if batch_node.count(x) == 0: 
            h_node_two.append(torch.full([50, 128], 0).to(device))
            h_node_three.append(torch.full([50, 128], 0).to(device))
            in_degree_2d_final.append(torch.full([50], 0).to(device))
            out_degree_2d_final.append(torch.full([50], 0).to(device))
            dist_3d_final.append(torch.full([50], 0).to(device))
            mask.append([] + 50* [-100000])
            continue

        mask.append([] + batch_node.count(x) * [0] + (50 - batch_node.count(x)) * [-100000])
        mask_out.append([] + batch_node.count(x) * [1] + (50 - batch_node.count(x)) * [0])
        
        oral_node_2d = emb_2d[flag:flag + batch_node.count(x)]
        oral_node_3d = emb_3d[flag:flag + batch_node.count(x)]
        if oral_node_2d.size() != oral_node_3d.size():
            print("not equal!!!")
            print(oral_node_2d.size(), oral_node_3d.size())
        in_degree_2d_oral = in_degree_2d[flag:flag + batch_node.count(x)]
        out_degree_2d_oral = out_degree_2d[flag:flag + batch_node.count(x)]
        dist_3d_oral = dist_3d[flag:flag + batch_node.count(x)]

        flag += batch_node.count(x)
        
        temp_2d = torch.full([(50-oral_node_2d.size()[0]), 128], 0).to(device)
        temp_3d = torch.full([(50-oral_node_3d.size()[0]), 128], 0).to(device)
        temp_in_degree_2d = torch.full([(50-oral_node_2d.size()[0])], 0).to(device)
        temp_out_degree_2d = torch.full([(50-oral_node_2d.size()[0])], 0).to(device)
        temp_out_dist_3d = torch.full([(50-oral_node_3d.size()[0])], 0).to(device)

        final_node_2d = torch.cat((oral_node_2d, temp_2d),0)
        final_node_3d = torch.cat((oral_node_3d, temp_3d),0)
        in_degree_2d_oral = torch.cat((in_degree_2d_oral, temp_in_degree_2d),0)
        out_degree_2d_oral = torch.cat((out_degree_2d_oral, temp_out_degree_2d),0)
        dist_3d_oral = torch.cat((dist_3d_oral, temp_out_dist_3d),0)

        if(dist_3d_oral.size()[-1] != 50):
            print(dist_3d_oral.size()[-1])
            print(dist_3d.size(), emb_3d.size(), batch.size())
            print("erro! run again!")

        h_node_two.append(final_node_2d)
        h_node_three.append(final_node_3d)
        in_degree_2d_final.append(in_degree_2d_oral)
        out_degree_2d_final.append(out_degree_2d_oral)
        dist_3d_final.append(dist_3d_oral)


    h_node_two = torch.stack(h_node_two).to(device)
    h_node_three = torch.stack(h_node_three).to(device)
    in_degree_2d_final = torch.stack(in_degree_2d_final).to(device)
    out_degree_2d_final = torch.stack(out_degree_2d_final).to(device)
    dist_3d_final = torch.stack(dist_3d_final).to(device)
    mask_2d = torch.tensor(mask, dtype=torch.float)
    mask_2d =  mask_2d.to(device).unsqueeze(1).unsqueeze(2)
    mask_2d_out = torch.tensor(mask_out, dtype=torch.float).to(device)
    mask_3d = torch.tensor(mask, dtype=torch.float)
    mask_3d =  mask_3d.to(device).unsqueeze(1).unsqueeze(2)
    mask_3d_out = torch.tensor(mask_out, dtype=torch.float).to(device)

    # 2d transformer emb
    emb_2d_high, attention_scores_2d = model_T_2d(h_node_two, mask_2d, True, in_degree_2d=in_degree_2d_final, out_degree_2d=out_degree_2d_final, spe=spe) # attention_scores_2d: [batch, head, node_len, node_len]
    # 3d transformer emb
    emb_3d_high, attention_scores_3d = model_T_3d(h_node_three, mask_3d, False, dist_3d=dist_3d_final, dist_m=dist_m)
    attention_scores_2d = torch.mean(attention_scores_2d, dim=-2) # every node
    attention_scores_2d = torch.mean(attention_scores_2d, dim=-2) # every head
    attention_scores_3d = torch.mean(attention_scores_3d, dim=-2) # every node
    attention_scores_3d = torch.mean(attention_scores_3d, dim=-2) # every head

    emb_2d_high = torch.div(torch.sum(torch.mul(emb_2d_high ,mask_2d_out.unsqueeze(2)),dim=1), torch.sum(mask_2d_out, dim=1).unsqueeze(1))
    emb_3d_high = torch.div(torch.sum(torch.mul(emb_3d_high ,mask_3d_out.unsqueeze(2)),dim=1), torch.sum(mask_3d_out, dim=1).unsqueeze(1))


    return emb_2d_high, emb_3d_high, attention_scores_2d, attention_scores_3d

# Generate 2D and 3D attention-guided mask with pior knowledge
def get_mask_indices(attention_scores, x, batch_node): # x: batch.x ; batch_node: record the number of each graph's node

    # rank attention
    x_c = x.clone().detach()
    attention_sorted, rank_indices = torch.sort(attention_scores, dim=1, descending=True)
    count = 0
    mask_list = []
    temp = []

    for i in range(batch_node[len(batch_node)-1] + 1):
        size = batch_node.count(i)
        temp = rank_indices[i][:size] # get true node; from high to low        
        # attention-guided mask top
        temp = (temp[:int(size * MASK_RATIO)] + count).tolist()
        count += size
        mask_list.extend(temp)
    
    # mask atom
    x_c[mask_list,0] = torch.tensor(119).cuda()
    x_c[mask_list,1] = torch.tensor(5).cuda()
    return x

# CL framework
class CL_model(nn.Module):
    def __init__(self):
        super(CL_model, self).__init__()
        self.model_2d = GNN()
        self.model_3d= SchNet()
        self.model_T_2d = Transformer_E()
        self.model_T_3d = Transformer_E()
        self.model_1d = transformer_1d()
        self.projection_2d_low = projection_head()
        self.projection_2d_high = projection_head()
        self.projection_3d_low = projection_head()
        self.projection_3d_high = projection_head()
        self.projection_1d_low = projection_head().to(device)
        self.projection_1d_high = projection_head().to(device)
        
    def forward(self, batch_nol):
        batch_nol = batch_nol.to(device)
        
        preprocessor.process(batch_nol)
        
        # 1d normal
        smiles_emb, smi_mask = torch.from_numpy(np.asarray(batch_nol.smiles)), torch.from_numpy(np.asarray(batch_nol.mask))
        emb_1d, attention_scores_1d = self.model_1d(torch.from_numpy(np.asarray(smiles_emb)), torch.from_numpy(np.asarray(smi_mask)))
        attention_scores_1d = torch.mean(attention_scores_1d, dim=-2) # every node
        attention_scores_1d = torch.mean(attention_scores_1d, dim=-2) # every head
        
        emb_1d_low = emb_1d[3]
        emb_1d_high = emb_1d[7]
        
        # 1d mask
        new_x_1d = get_mask_indices_1d(attention_scores_1d, batch_nol.smiles, smi_mask)
        emb_1d_mask, _ = self.model_1d(torch.from_numpy(np.asarray(new_x_1d)), torch.from_numpy(np.asarray(smi_mask)))
        emb_1d_low_mask = emb_1d_mask[3]
        emb_1d_high_mask = emb_1d_mask[7]
        
        # 1d get mean
        mask_1d_out = smi_mask.to(device)
        emb_1d_low = torch.div(torch.sum(torch.mul(emb_1d_low ,mask_1d_out.unsqueeze(2)),dim=1), torch.sum(mask_1d_out, dim=1).unsqueeze(1))
        emb_1d_high = torch.div(torch.sum(torch.mul(emb_1d_high ,mask_1d_out.unsqueeze(2)),dim=1), torch.sum(mask_1d_out, dim=1).unsqueeze(1))
        emb_1d_low_mask = torch.div(torch.sum(torch.mul(emb_1d_low_mask ,mask_1d_out.unsqueeze(2)),dim=1), torch.sum(mask_1d_out, dim=1).unsqueeze(1))
        emb_1d_high_mask = torch.div(torch.sum(torch.mul(emb_1d_high_mask ,mask_1d_out.unsqueeze(2)),dim=1), torch.sum(mask_1d_out, dim=1).unsqueeze(1))


        # 2d background
        in_degree_2d = degree(batch_nol.edge_index[0], num_nodes=len(batch_nol.x)).int()
        out_degree_2d = degree(batch_nol.edge_index[1], num_nodes=len(batch_nol.x)).int()
        spe_nol = get_spatil_pos(batch_nol.batch, batch_nol.edge_index, batch=batch_nol)

        # 3d background
        dist_3d, dist_m_nol = get_3d_pos(batch_nol.pos, batch_nol.batch)

        
        # 2d graph emb
        emb_2d_low_nol, emb_2d_low_nol_res, _ = self.model_2d(batch_nol.x, batch_nol.edge_index, batch_nol.edge_attr)
        emb_2d_low_nol_out = molecule_readout_func(emb_2d_low_nol, batch_nol.batch)
        # residue
        emb_2d_low_nol = emb_2d_low_nol + emb_2d_low_nol_res

        # 3d graph emb
        emb_3d_low_nol, emb_3d_low_nol_res,_ = self.model_3d(batch_nol.x[:,0], batch_nol.pos, batch_nol.batch)
        emb_3d_low_nol_out = molecule_readout_func(emb_3d_low_nol, batch_nol.batch)
        emb_3d_low_nol = emb_3d_low_nol + emb_3d_low_nol_res

        # 2d 3d transfomer emb
        emb_2d_high_nol, emb_3d_high_nol, attention_scores_2d, attention_scores_3d = get_transformer_emb(batch_nol.batch, emb_2d_low_nol, emb_3d_low_nol, in_degree_2d, out_degree_2d, dist_3d, dist_m_nol, spe_nol, self.model_T_2d, self.model_T_3d)
           
           
        # 2d mask node                
        new_x_2d = get_mask_indices(attention_scores_2d,batch_nol.x, batch_nol.batch.tolist() )

        
        # 2d mask emb
        emb_2d_low_mask, emb_2d_low_mask_res, _ = self.model_2d(new_x_2d, batch_nol.edge_index, batch_nol.edge_attr)
        emb_2d_low_mask_out = molecule_readout_func(emb_2d_low_mask, batch_nol.batch)

        emb_2d_low_mask = emb_2d_low_mask + emb_2d_low_mask_res
        
        # 3d mask   
        new_x_3d = get_mask_indices(attention_scores_3d, batch_nol.x, batch_nol.batch.tolist())
        
        # 3d mask emb
        emb_3d_low_mask, emb_3d_low_mask_res,_ = self.model_3d(new_x_3d[:,0], batch_nol.pos, batch_nol.batch)
        emb_3d_low_mask_out = molecule_readout_func(emb_3d_low_mask, batch_nol.batch)
        emb_3d_low_mask = emb_3d_low_mask + emb_3d_low_mask_res

        # 2d 3d mask emb
        emb_2d_high_mask, emb_3d_high_mask, _, _ = get_transformer_emb(batch_nol.batch, emb_2d_low_mask, emb_3d_low_mask, in_degree_2d, out_degree_2d, dist_3d, dist_m_nol, spe_nol, self.model_T_2d, self.model_T_3d)   # without x, so it's ok

        emb_2d_low_mask = emb_2d_low_mask_out
        emb_3d_low_mask = emb_3d_low_mask_out

        # proj head
        emb_2d_low_nol_out = self.projection_2d_low(emb_2d_low_nol_out)
        emb_3d_low_nol_out = self.projection_3d_low(emb_3d_low_nol_out)
        emb_2d_high_nol = self.projection_2d_high(emb_2d_high_nol)
        emb_3d_high_nol = self.projection_3d_high(emb_3d_high_nol)
        emb_2d_low_mask_out = self.projection_2d_low(emb_2d_low_mask_out)
        emb_3d_low_mask_out = self.projection_3d_low(emb_3d_low_mask_out)
        emb_2d_high_mask = self.projection_2d_high(emb_2d_high_mask)
        emb_3d_high_mask = self.projection_3d_high(emb_3d_high_mask)
        emb_1d_low_out = self.projection_1d_low(emb_1d_low)
        emb_1d_high_out = self.projection_1d_high(emb_1d_high)
        emb_1d_low_mask_out = self.projection_1d_low(emb_1d_low_mask)
        emb_1d_high_mask_out = self.projection_1d_high(emb_1d_high_mask)
        
        return emb_2d_low_nol_out, emb_3d_low_nol_out, emb_2d_high_nol, emb_3d_high_nol,  \
                emb_2d_low_mask_out, emb_3d_low_mask_out, emb_2d_high_mask, emb_3d_high_mask, \
                emb_1d_low_out, emb_1d_high_out, emb_1d_low_mask_out, emb_1d_high_mask_out

# Projection head
class projection_head(nn.Module):
    def __init__(self):
        super(projection_head, self).__init__()
        self.line1 = nn.Linear(128, 128)
        self.line2 = nn.Linear(128, 64)


        self.relu = nn.ReLU()
    
    def forward(self, emb):
        out = self.line1(emb)
        out = self.relu(out)
        out = self.line2(out)

        return out


def save_model(save_best, epoch):
    if save_best:
        saver_dict = {
            'cl_model': cl_model.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        torch.save(saver_dict, output_model_dir + 'model_'+str(epoch)+'.pth')

    else:
        saver_dict = {

            'cl_model': cl_model.state_dict(),
        }
        torch.save(saver_dict, output_model_dir + 'model_final.pth')
    return

def train(train_loader_nol, optimizer,data_len):
    
    train_loader = train_loader_nol

    cl_model.train()

    
    CL_multi_low_total = 0
    CL_multi_high_total = 0

    
    for step, batch_nol in enumerate(train_loader):
        optimizer.zero_grad()

        temp = cl_model(batch_nol)

        emb_2d_low_nol_out, emb_3d_low_nol_out, emb_2d_high_nol, emb_3d_high_nol,  \
                emb_2d_low_mask_out, emb_3d_low_mask_out, emb_2d_high_mask, emb_3d_high_mask, \
                emb_1d_low_out, emb_1d_high_out, emb_1d_low_mask_out, emb_1d_high_mask_out = temp
        
        #low
        emb_1d_low_views = torch.cat((emb_1d_low_out.unsqueeze(1), emb_1d_low_mask_out.unsqueeze(1)), 1)
        emb_2d_low_views = torch.cat((emb_2d_low_nol_out.unsqueeze(1), emb_2d_low_mask_out.unsqueeze(1)), 1)
        emb_3d_low_views = torch.cat((emb_3d_low_nol_out.unsqueeze(1), emb_3d_low_mask_out.unsqueeze(1)), 1)
        label_low = torch.tensor(np.array([il for il in range(emb_2d_low_nol_out.size()[0])])).float().to(device)
        emb_low_features = torch.cat((emb_2d_low_views, emb_3d_low_views,emb_1d_low_views), 1)

        CL_multi_low = AMC_loss_low(emb_low_features, epoch, label_low, train=True)
        
        # high
        emb_1d_high_views = torch.cat((emb_1d_high_out.unsqueeze(1), emb_1d_high_mask_out.unsqueeze(1)), 1)
        emb_2d_high_views = torch.cat((emb_2d_high_nol.unsqueeze(1), emb_2d_high_mask.unsqueeze(1)), 1)
        emb_3d_high_views = torch.cat((emb_3d_high_nol.unsqueeze(1), emb_3d_high_mask.unsqueeze(1)), 1)
        label_high = torch.tensor(np.array([ih for ih in range(emb_2d_high_nol.size()[0])])).float().to(device)
        emb_high_features = torch.cat((emb_2d_high_views, emb_3d_high_views,emb_1d_high_views), 1)

        CL_multi_high = AMC_loss_high(emb_high_features, epoch, label_high, train=True)
        
    
        
        CL_multi_low_total += CL_multi_low.detach().cpu()
        CL_multi_high_total += CL_multi_high.detach().cpu()
        

        loss = 0
        loss = CL_multi_low + CL_multi_high
        loss.backward()
        optimizer.step()
    global optimal_loss

    CL_multi_low_total /= data_len
    CL_multi_high_total /= data_len
    print((AMC_loss_low.last_loss[epoch] / data_len))
    print((AMC_loss_high.last_loss[epoch] / data_len))
    temp_loss = CL_multi_low_total + CL_multi_high_total

    if temp_loss < optimal_loss:
        optimal_loss = temp_loss

    print('CL low Loss: {:.5f}\t\tCL high Loss: {:.5f}\t'.format(
        CL_multi_low_total, CL_multi_high_total))
    
    print('temp loss: ', temp_loss,' optim loss: ', optimal_loss)
    
    return

def evaluate(valid_loader_nol, val_data_len):
    valid_loader = valid_loader_nol

    cl_model.eval()
    
    CL_multi_low_total = 0
    CL_multi_high_total = 0

    
    with torch.no_grad():
        for step, batch_nol in enumerate(valid_loader):
            try:
                temp = cl_model(batch_nol) 
            except:
                continue
            emb_2d_low_nol_out, emb_3d_low_nol_out, emb_2d_high_nol, emb_3d_high_nol,  \
                emb_2d_low_mask_out, emb_3d_low_mask_out, emb_2d_high_mask, emb_3d_high_mask, \
                emb_1d_low_out, emb_1d_high_out, emb_1d_low_mask_out, emb_1d_high_mask_out = temp        
           
           

            # multi positive
            #low
            emb_1d_low_views = torch.cat((emb_1d_low_out.unsqueeze(1), emb_1d_low_mask_out.unsqueeze(1)), 1)
            emb_2d_low_views = torch.cat((emb_2d_low_nol_out.unsqueeze(1), emb_2d_low_mask_out.unsqueeze(1)), 1)
            emb_3d_low_views = torch.cat((emb_3d_low_nol_out.unsqueeze(1), emb_3d_low_mask_out.unsqueeze(1)), 1)

            
            
            label_low = torch.tensor(np.array([il for il in range(emb_2d_low_nol_out.size()[0])])).float().to(device)
            emb_low_features = torch.cat((emb_2d_low_views, emb_3d_low_views,emb_1d_low_views), 1)
            CL_multi_low = AMC_loss_low(emb_low_features, epoch, label_low, train=False)
            # high
            emb_1d_high_views = torch.cat((emb_1d_high_out.unsqueeze(1), emb_1d_high_mask_out.unsqueeze(1)), 1)
            emb_2d_high_views = torch.cat((emb_2d_high_nol.unsqueeze(1), emb_2d_high_mask.unsqueeze(1)), 1)
            emb_3d_high_views = torch.cat((emb_3d_high_nol.unsqueeze(1), emb_3d_high_mask.unsqueeze(1)), 1)
            
        
            
            label_high = torch.tensor(np.array([ih for ih in range(emb_2d_high_nol.size()[0])])).float().to(device)
            emb_high_features = torch.cat((emb_2d_high_views, emb_3d_high_views,emb_1d_high_views), 1)
            CL_multi_high = AMC_loss_high(emb_high_features, epoch, label_high, train=False)
        


            CL_multi_low_total += CL_multi_low.detach().cpu()
            CL_multi_high_total += CL_multi_high.detach().cpu()

        
        CL_multi_low_total /= val_data_len
        CL_multi_high_total /= val_data_len


        temp_loss = CL_multi_low_total + CL_multi_high_total

            
        print('CL low Loss: {:.5f}\t\tCL high Loss: {:.5f}\t'.format(
            CL_multi_low_total, CL_multi_high_total))


        print('valid loss: ', temp_loss)

    return temp_loss

class MyDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        xC = self.datasetC[index]
        return xA, xB, xC
    
    def __len__(self):
        return len(self.datasetA)


class PreprocessBatch:
    def process(self, batch):
        pos = batch.pos
        batch_node = batch.batch.tolist()
        pos_mean = global_mean_pool(pos, batch.batch)

        flag = 0
        num = []
        for x in range(batch_node[len(batch_node)-1] + 1):
            flag = batch_node.count(x)
            num.append(flag)
        pos = pos - torch.repeat_interleave(pos_mean, torch.tensor(num).to(device), dim=0)
        batch.pos = pos

if __name__ == '__main__':

    # pretraing dataset 
    dataset = PCQM4Mv2Dataset()
    split_idx = dataset.get_idx_split()



    randperm = torch.randperm(len(split_idx["train"]))
    train_idxs = randperm[: int((0.96) * len(split_idx["train"]))]
    dev_idxs = randperm[int(0.96 * len(split_idx["train"])) :]
    dataset_nol_train = dataset[train_idxs]
    
    train_sampler = SubsetRandomSampler(train_idxs)
    valid_sampler = SubsetRandomSampler(dev_idxs)


    train_loader_nol = torch_geometric.loader.DataListLoader(
        dataset_nol_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )
    valid_loader_nol = torch_geometric.loader.DataListLoader(
        dataset[dev_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )

    data_len = len(train_loader_nol)
    val_data_len = len(valid_loader_nol)
    print('train dataset length: ', data_len)
    print('val dataset length: ', val_data_len)
    
    
    # model
    cl_model = CL_model()
    cl_model = torch_geometric.nn.DataParallel(cl_model.cuda(), device_ids=device_ids)


    model_param_group = []
    model_param_group.append({'params': cl_model.parameters(), 'lr': 0.0001 * 1})

    optimizer = optim.Adam(model_param_group, weight_decay=1e-5)
    optimal_loss = 1e10
    
    preprocessor = PreprocessBatch()

    best_valid_loss = 10000


    for epoch in range(1, EPOCH + 1):
        print('Epoch: {}'.format(epoch))
        print("Training")
        train(train_loader_nol, optimizer,data_len)
        print("Evaluating")
        valid_loss = evaluate(valid_loader_nol, val_data_len)
        best_valid_loss = valid_loss
        save_model(True, epoch)
        print("  ")
    save_model(False, epoch)