import os
import os.path as osp
import shutil
from .MPP.utils.torch_util import replace_numpy_with_torchtensor
from .MPP.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import re
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdmolfiles import SDMolSupplier
from torch_geometric.data import InMemoryDataset
from .MPP.utils.features import atom_to_feature_vector, bond_to_feature_vector
import warnings
warnings.filterwarnings('error') 
import codecs
from subword_nmt.apply_bpe import BPE
bad_case = []

def drug2emb_encoder(smile):
    vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)

def floyd_warshall(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    adj_mat_copy = adjacency_matrix.astype(float, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    M = adj_mat_copy
    # path = np.zeros([n, n], dtype=np.int64)

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





class PCQM4Mv2Dataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset/",
        # smiles2graph=smiles2graphwithface,
        transform=None,
        pre_transform=None,
        xyzdir='dataset/pcqm4m-v2/pcqm4m-v2_xyz',
        mask_ratio = 0.5
    ):
        self.original_root = root
        self.mask_ratio = mask_ratio
        # self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        self.url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"

        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        self.xyzdir = xyzdir

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        print("no mask")
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        split_dict = self.get_idx_split()
        train_idxs = split_dict["train"].tolist()
        print("Converting SMILES strings into graphs...")
        data_list = []

        for i in tqdm(range(len(smiles_list))):
            # data = DGData()
            data = Data()
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
           
            if i in train_idxs:
                prefix = i // 10000
                prefix = "{0:04d}0000_{0:04d}9999".format(prefix)
                xyzfn = osp.join(self.xyzdir, prefix, f"{i}.sdf")
                mol = next(SDMolSupplier(xyzfn))
                pos = mol.GetConformer(0).GetPositions() #坐标 R
                num_atoms = mol.GetNumAtoms() # 数量 Z
            else:
                mol = Chem.MolFromSmiles(smiles)
                num_atoms = mol.GetNumAtoms()
                pos = np.zeros((num_atoms, 3), dtype=float)
            # atoms
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            x = np.array(atom_features_list, dtype=np.int64)
            # bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)
            
            
            data.x = torch.from_numpy(x).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            data.pos = torch.from_numpy(pos).to(torch.float)
            data.smiles, data.mask = drug2emb_encoder(smiles)
            data.ori_smiles = smiles

            if data.pos.size()[0] == 0 or data.pos.size()[1] == 0:
                print("zero!")
                print(data.pos.size())
                continue
            data.num_nodes = num_atoms
      

            data_list.append(data)

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict
