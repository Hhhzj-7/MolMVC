from torch_geometric.data import InMemoryDataset
import shutil, os
import os.path as osp
import torch
import re
from torch_sparse import SparseTensor

import numpy as np
from tqdm import tqdm
from ...utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
from .deepchem_dataloader import (
    load_molnet_dataset,
    get_task_type,
)
from copy import deepcopy
import codecs
from subword_nmt.apply_bpe import BPE
import pandas as pd
from ....MPP.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data


class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0

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

class DCGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None):
        assert name.startswith("dc-")
        name = name[len("dc-") :]
        self.name = name
        self.dirname = f"dcgraphproppred_{name}"
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        print(self.root)
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices, self._num_tasks = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        path = os.path.join(self.root, "split", "split_dict.pt")
        return torch.load(path)

    @property
    def task_type(self):
        return get_task_type(self.name)

    @property
    def eval_metric(self):
        return "rocauc" if "classification" in self.task_type else "mae"

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        pass

    def process(self):
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        _, dfs, _ = load_molnet_dataset(self.name)

        num_tasks = len(dfs[0]["labels"].values[0])

        for insert_idx, df in zip([train_idx, valid_idx, test_idx], dfs):
            smiles_list = df["text"].values.tolist()
            labels_list = df["labels"].values.tolist()
            assert len(smiles_list) == len(labels_list)

            for smiles, labels in zip(smiles_list, labels_list):
                data = DGData()
                mol = Chem.MolFromSmiles(smiles)
                graph = smiles2graphwithface(mol)

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_nodes__ = int(graph["num_nodes"])

                if "classification" in self.task_type:
                    data.y = torch.as_tensor(labels).view(1, -1).to(torch.long)
                else:
                    data.y = torch.as_tensor(labels).view(1, -1).to(torch.float32)
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
                data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
                data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
                
                data.smiles_ori = smiles
                data.smiles, data.mask = drug2emb_encoder(smiles)


                data_list.append(data)
                insert_idx.append(len(data_list))
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices, num_tasks), self.processed_paths[0])

        os.makedirs(osp.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.as_tensor(train_idx, dtype=torch.long),
                "valid": torch.as_tensor(valid_idx, dtype=torch.long),
                "test": torch.as_tensor(test_idx, dtype=torch.long),
            },
            osp.join(self.root, "split", "split_dict.pt"),
        )
        
class get_emb_Dataset(InMemoryDataset):
    def __init__(self, root="dataset_pre", transform=None, pre_transform=None, smiles_list=None):
        self.smiles_list = smiles_list
        super().__init__(root, transform, pre_transform, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    # def download(self):
    #     pass

    def process(self):

        file_path = self.smiles_list
        print(file_path)

        df = pd.read_csv(file_path, header=None, names=["ID", "SMILES"])

        data_list = []

        for index, row in df.iterrows():
            data = DGData()
            mol = Chem.MolFromSmiles(row['SMILES'])
            graph = smiles2graphwithface(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])

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
            data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            
            data.smiles_ori = row['SMILES']
            data.smiles, data.mask = drug2emb_encoder(row['SMILES'])
            data.id = row['ID']

            data_list.append(data)


        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = DCGraphPropPredDataset("dc-bbbp")
    split_index = dataset.get_idx_split()
    print(split_index)

