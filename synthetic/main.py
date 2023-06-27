from dataset import (
    CSL,
    EXP,
    SR25
    )

from networks.model import PathNN
from utils.loader import prepare_loaders
from utils.logger import PrinterLogger
from utils.stopper import Patience

import numpy as np
import datetime
import logging
import argparse
import os, json, pickle
import torch
from torch import nn
import tqdm 
from collections import defaultdict
from torch_geometric.loader import DataLoader


DATASETS = {
    "CSL" : CSL.CSL,
    "EXP" : EXP.EXP,
    "CEXP" : EXP.EXP,
    "SR25" : SR25.SR25,
}
SR25_NAMES = [
    'sr16622.g6',
    'sr251256.g6',
    'sr261034.g6',
    'sr281264.g6',
    'sr291467.g6',
    'sr361446.g6',
    'sr401224.g6'
]
DATASET_NAMES = {
    'SR25' : SR25_NAMES,
    'CSL' : ['CSL'],
    'EXP' : ['EXP'],
    'CEXP' : ['CEXP']
}

def train(model, data, criterion, optimizer):
    output = model(data)
    loss_train = criterion(output, data.y)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    return output, loss_train

@torch.no_grad()
def test(model, data, criterion):
    output = model(data)
    loss_test = criterion(output, data.y)
    return output, loss_test

def train_class(model, loaders, epochs = 200, patience = 100, lr = 1e-2, r = None, print_errors = False) :

    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = Patience(patience = patience, use_loss = False, save_path=os.path.join("models/", args.dataset,"model"+now+".pth.tar"))
    
    criterion = nn.CrossEntropyLoss()
    early_stopper.val_acc = 0
    pbar_train = tqdm.tqdm(range(epochs), desc = "Epoch 0 Loss 0")
    for epoch in pbar_train:
        model.train()
        train_loss = 0 
        train_correct = 0 
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            output, loss = train(model, data, criterion,optimizer)
            train_loss += loss.item() * data.num_graphs
            preds = output.max(1)[1].type_as(data.y)
            train_correct += torch.sum(preds.eq(data.y.double())).item()

        val_loss = 0
        val_correct = 0 
        all_preds, ys = [], []
        model.eval()
        for idx, data in enumerate(val_loader):
            output, loss = test(model, data.to(device), criterion)
            val_loss += loss.item() * data.num_graphs
            preds = output.max(1)[1].type_as(data.y)
            val_correct += torch.sum(preds.eq(data.y.double())).item()
            all_preds.append(preds)
            ys.append(data.y)

        val_acc = val_correct/len(val_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct/len(train_loader.dataset)

        if all([val_acc > early_stopper.val_acc, print_errors]) : 
            print(torch.cat([
                torch.cat(all_preds).unsqueeze(1),
                torch.cat(ys).unsqueeze(1),

            ], dim = 1))

        if early_stopper.stop(epoch, val_loss, val_acc, model = model) : 
            break

        pbar_train.set_description(f"Epoch {epoch+1} Train Loss {train_loss:0.3f} Train Acc {train_acc:0.3f} Val loss {round(val_loss,3)} Val acc {round(val_acc,3)} Best Val Loss {round(early_stopper.val_loss, 3)} Best Val Acc {early_stopper.val_acc:0.3f}")

    checkpoint = torch.load(os.path.join("models/", args.dataset,"model"+now+".pth.tar"))
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    test_count, test_correct= 0,0
    val_count, val_correct = 0,0
    train_count, train_correct = 0,0

    model.eval()

    for idx, data in enumerate(train_loader):
        output, loss = test(model, data.to(device),criterion)
        train_count += output.size(0)
        preds = output.max(1)[1].type_as(data.y)
        train_correct += torch.sum(preds.eq(data.y.double())).item()

    for idx, data in enumerate(val_loader):
        output, loss = test(model, data.to(device),criterion)
        val_count += output.size(0)
        preds = output.max(1)[1].type_as(data.y)
        val_correct += torch.sum(preds.eq(data.y.double())).item()

    for idx, data in enumerate(test_loader):
        output, loss = test(model, data.to(device),criterion)
        test_count += output.size(0)
        preds = output.max(1)[1].type_as(data.y)
        test_correct += torch.sum(preds.eq(data.y.double())).item()

    log.print_and_log(f"Train Acc {(train_correct/train_count):0.2f} Val Acc {(val_correct/val_count):0.2f} Test Acc {(test_correct/test_count):0.2f}")

    return train_correct/train_count, val_correct/val_count, test_correct/test_count


def main_class(dataset, model_config, train_config) : 
    splits = dataset.get_all_splits_idx()
    results = defaultdict(list)
    n_splits = dataset.n_splits
    model = PathNN(device=device, **model_config).to(device)

    for index in range(n_splits) : 
        model.reset_parameters()
        log.print_and_log(f'Split {index+1}/{n_splits}')

        train_loader, val_loader, test_loader = prepare_loaders(
            dataset, 
            splits, 
            batch_size = train_config["batch_size"], 
            index = index)

        train_acc, val_acc, test_acc = train_class(
            model, 
            (train_loader, val_loader, test_loader), 
            epochs = train_config["epochs"],
            patience = train_config["patience"],
            lr = train_config["lr"])
        
        results["train"].append(train_acc)
        results["val"].append(val_acc)
        results["test"].append(test_acc)

    return {k:np.asarray(v) for k,v in results.items()} 

def _isomorphism(preds, eps = 1e-5, p = 2):
    # NB: here we return the failure percentage... the smaller the better!
    assert preds is not None
    #assert preds.dtype == np.float64
    preds = torch.tensor(preds, dtype=torch.float64)
    mm = torch.pdist(preds, p=p)
    wrong = (mm < eps).sum().item()
    metric = wrong / mm.shape[0]
    return metric


def main_iso(dataset, model_config, train_config) :
    loader = DataLoader(dataset,
                        batch_size= train_config["batch_size"],
                        shuffle=False)
    model = PathNN(predict = False, device=device, **model_config).to(device)
    res = []
    for i in range(5) : 
        torch.manual_seed(i)
        model.reset_parameters()
        embeddings, lst = [], []
        model.eval()
        for data in tqdm.tqdm(loader) : 
            pre = model(data.to(device))
            embeddings.append(pre.detach().cpu())
        print(f'Failure Rate : {_isomorphism(torch.cat(embeddings, 0).detach().cpu().numpy())}')
        res.append(_isomorphism(torch.cat(embeddings, 0).detach().cpu().numpy()))

    return {'n_sim_pairs' : np.asarray(res)}


MAIN_FUNC = {"iso" : main_iso, "classification" : main_class}


if __name__=="__main__" : 


    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=str, default="cuda",
                        help='choose between cuda or cpu')
    parser.add_argument('--dataset', type=str, default="CSL",
                        help='Experimental dataset name (default: CSL)')
    parser.add_argument('--path-type', type=str, default="all",
                        help='Which path-type to consider. Default to all ')


    args = parser.parse_args()

    #Return to synthetic directory in case we start script for script directory
    if os.getcwd()[-9:] != 'synthetic' : 
        os.chdir(os.path.dirname(os.getcwd()) )

    if not os.path.exists(os.path.join(os.getcwd(), 'results', args.dataset)) : 
        os.makedirs(os.path.join(os.getcwd(), 'results', args.dataset))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs', args.dataset)) : 
        os.makedirs(os.path.join(os.getcwd(), 'logs', args.dataset))
    if not os.path.exists(os.path.join('models', args.dataset)) : 
        os.makedirs(os.path.join('models', args.dataset))
        
    device = torch.device(args.device)

    now = "_" + "-".join(str(datetime.datetime.today()).split()).split('.')[0].replace(':','.')
    logging.basicConfig(filename=os.path.join('logs', args.dataset, args.dataset+now+'.log'), level=logging.INFO, filemode="w")
    log = PrinterLogger(logging.getLogger(__name__) )

    with open(os.path.join("configs", args.dataset + "_config.json"), "r") as f :
        config = json.load(f)
    
    results = defaultdict(dict)
    path_types = [args.path_type] if not args.path_type == "all"  else ["shortest_path", "all_shortest_paths", "all_simple_paths"]
    
    
    for path_type in path_types : 
        if config['task'] == 'iso' : 
            torch.set_default_dtype(torch.float64)

        log.print_and_log(f"PROCESSING {path_type.upper()}")
        for d_name in DATASET_NAMES[args.dataset] : 
            log.print_and_log(f"Model Name {d_name.upper()}")
            if args.dataset == 'SR25' : 
                config[path_type]["dataset_config"]['dataset_name'] = d_name
            dataset = DATASETS[args.dataset](**config[path_type]["dataset_config"])
            results[path_type][d_name] = MAIN_FUNC[config["task"]](
                dataset,
                config[path_type]["model_config"],
                config[path_type]["train_config"])
            
            if config["task"] == "classification" : 
                log.print_and_log(f"Train Avg {results[path_type][d_name]['train'].mean():0.4f} Train Std {results[path_type][d_name]['train'].std():0.4f}")
                log.print_and_log(f"Val Avg   {results[path_type][d_name]['val'].mean():0.4f} Val Std   {results[path_type][d_name]['val'].std():0.4f}")
                log.print_and_log(f"Test Avg  {results[path_type][d_name]['test'].mean():0.4f} Test Std  {results[path_type][d_name]['test'].std():0.4f}")
            
            elif config["task"] == "iso" : 
                log.print_and_log(f"Avg # of similar pairs: {results[path_type][d_name]['n_sim_pairs'].mean()}")
                log.print_and_log(f"Std # of similar pairs: {results[path_type][d_name]['n_sim_pairs'].std()}")


        with open(os.path.join("results", args.dataset+ "_results.pkl"),"wb") as f : 
            pickle.dump(results, f)