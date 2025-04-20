
import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch import nn
from tqdm import tqdm
import numpy as np
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from downstream_until import CL_model
from process_dataset.MPP.data.DCGraphPropPredDataset.dataset import DCGraphPropPredDataset, get_emb_Dataset
from process_dataset.MPP.utils.dist import init_distributed_mode
from process_dataset.MPP.utils.evaluate import Evaluator
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

Folder_1d = 'emb_1d/'
Folder_2d = 'emb_2d/'
Input_SMILES = 'example_drug_SMILES.csv' # id, SMILES
Data_root = 'data_pre'
# random seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, device, train_loader, optimizer, args):
    model.train()
    loss_accum = 0
    pbar = tqdm(train_loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_attrs, out_1d, out_2d = model(batch)
            is_labeled = batch.y == batch.y


            loss = F.binary_cross_entropy_with_logits(
                pred_attrs.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
            )

            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item()

    return {"loss": loss_accum / (step + 1)}

def save_emb(model, device, train_loader):
    model.train()
    pbar = tqdm(train_loader, desc="Iteration")

    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            continue
        print( batch.id, batch.x)
        _, out_1d, out_2d = model(batch, True)
        torch.save(out_1d, Folder_1d + str(batch.id[0]))
        torch.save(out_2d, Folder_2d + str(batch.id[0]))



def evaluate(model, device, loader, args, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    with torch.no_grad():
        for step, batch in enumerate(
            tqdm(loader, desc="Valid Iteration", disable=args.disable_tqdm)
        ):
            batch = batch.to(device)
            pred_attrs = model(batch)
            y_true.append(batch.y.view(pred_attrs.shape).detach().cpu())
            y_pred.append(pred_attrs.detach().cpu())
            total_preds = torch.cat((total_preds, pred_attrs.cpu()), 0)
            total_labels = torch.cat((total_labels, batch.y.cpu()), 0)
        

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)


    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)

    # id, SMILES
    dataset = get_emb_Dataset(root= Data_root,smiles_list= Input_SMILES)


    train_loader = DataLoader(
        dataset, num_workers=args.num_workers,batch_size=args.batch_size,shuffle=False,
    )

    seeds = [1]
    
    for seed in seeds:
        print("seed:",seed)
        set_seed(seed)
        args.disable_tqdm = True
        model = CL_model(device, True, True).to(device)

        save_emb(model, device, train_loader)

if __name__ == "__main__":
    main()

