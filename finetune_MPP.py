

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
from process_dataset.MPP.data.DCGraphPropPredDataset.dataset import DCGraphPropPredDataset
from process_dataset.MPP.utils.dist import init_distributed_mode
from process_dataset.MPP.utils.evaluate import Evaluator
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

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
            pred_attrs = model(batch)
            is_labeled = batch.y == batch.y


            loss = F.binary_cross_entropy_with_logits(
                pred_attrs.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
            )

            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item()

    return {"loss": loss_accum / (step + 1)}


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="dc-bbbp")


    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)

    if args.dataset.startswith("dc"):
        dataset  = DCGraphPropPredDataset(args.dataset)

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset, dataset=dataset)

    dataset_train = dataset[split_idx["train"]]


    train_loader = DataLoader(
        dataset_train, num_workers=args.num_workers,batch_size=args.batch_size,shuffle=False,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    

    seeds = [1]
    
    for seed in seeds:
        print("seed:",seed)
        set_seed(seed)
        args.disable_tqdm = True
        model = CL_model(device, True, True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=eval('1e-2'))

        valid_curve = []
        test_curve = []
        best_val_epoch = 0

        for epoch in range(1, args.epochs + 1):
        
        
            print("Epoch {}".format(epoch))
            print("Training...")
            train_loss_dict = train(
                model, device, train_loader, optimizer, args
            )
            print("Evaluating...")
            valid_perf = evaluate(model, device, valid_loader, args, evaluator)
            test_perf = evaluate(model, device, test_loader, args, evaluator)

            print("Validation", valid_perf, "Test", test_perf)
            
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])
            
            
            now_best_val_epoch = np.argmax(np.array(valid_curve))
            if best_val_epoch != now_best_val_epoch:
                best_val_epoch = now_best_val_epoch
            print("Test score: {}".format(test_curve[best_val_epoch]))



        best_val_epoch = np.argmax(np.array(valid_curve))

        print("Finished training!")
        print("seed:",seed)        
        print("Best validation score: {}".format(valid_curve[best_val_epoch]))
        print("Test score: {}".format(test_curve[best_val_epoch]))


if __name__ == "__main__":
    main()

