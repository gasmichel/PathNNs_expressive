import argparse
from importlib.util import module_for_loader
import networkx as nx
import numpy as np
import time
import pickle
import json
import scipy.sparse as sp
from math import ceil
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from random import randint
import logging
import os
import datetime
from collections import defaultdict

import torch
import torch.nn.init as init 
from torch.autograd import Variable 
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

from torch_geometric.utils.convert import from_networkx
#from tqdm import tqdm
import tqdm
#from tqdm.contrib.logging import logging_redirect_tqdm

from torch_geometric.nn import GCN, GIN

from model import *
from utils import *


# Argument parser
parser = argparse.ArgumentParser(description='PathNN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='Number of epochs to train')
parser.add_argument('--cutoff', type=int, default=3, metavar='N', help='Max number of nodes in paths (path length +1)')
parser.add_argument('--path-type', default='all_shortest_paths', help='Type of extracted path')
parser.add_argument('--patience', default=250, type = int, help='Number of patience rounds for early stopping')
parser.add_argument('--dataset-config-path', default='dataset_config.json', help='Path to config file containing which node initial representation to use.')
parser.add_argument('--device', default = "cuda", type = str, help='Whether to consider paths from node u to v AND paths from node v to u')
parser.add_argument('--residuals', default = False, action = 'store_true', help='Whether to use residual connection in the update equation.')


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

def main():
    
    args = parser.parse_args()
    device = torch.device("cuda") if all([args.device == "cuda", torch.cuda.is_available()]) else torch.device("cpu")

    #Return to synthetic directory in case we start script for script directory
    if os.getcwd()[-10:] != 'TUDatasets' : 
        os.chdir(os.path.dirname(os.getcwd()))

    with open(args.dataset_config_path, 'r') as f :
        config = json.load(f)[args.dataset]
    Gs, features, y, splits = load_data(args.dataset, config['use_node_labels'], config['use_node_attributes'], config['degree_as_tag'])

    features_dim = features[0].shape[1]
    n_classes = len(np.unique(y))
    criterion = torch.nn.CrossEntropyLoss()
    
    ###### LOGGING AND DIRECTORY ######
    if not os.path.exists(os.path.join('results', args.dataset)) : 
        os.makedirs(os.path.join('results', args.dataset))
    if not os.path.exists(os.path.join('logs', args.dataset)) : 
        os.makedirs(os.path.join('logs', args.dataset))
    if not os.path.exists(os.path.join('models', args.dataset)) : 
        os.makedirs(os.path.join('models', args.dataset))
    now = "_" + "-".join(str(datetime.datetime.today()).split()).split('.')[0].replace(':','.')
    residuals = '_residuals' if args.residuals else ''
    program_name = "cutoff_"+str(args.cutoff)+"_path_"+str(args.path_type)+f'_bs_{args.batch_size}'+residuals+now

    logging.basicConfig(filename=os.path.join('logs', args.dataset, program_name+'.log'), level=logging.INFO, filemode="w")
    log = PrinterLogger(logging.getLogger(__name__) )
    
    dataset = PathDataset(Gs, features, y, args.cutoff, args.path_type)

    accs_folds = list()
    if args.path_type == 'all_simple_paths' : 
        encode_distances = True 
    else : 
        encode_distances = False
        
    np.random.seed(10)
    torch.random.manual_seed(10) 
    for it in range(10) : 
        log.info('-'*30+f'ITERATION {str(it+1)}'+"-"*30)
        print('-'*30+f'ITERATION {str(it+1)}'+"-"*30)
        train_index = splits[it]['model_selection'][0]['train']
        val_index = splits[it]['model_selection'][0]['validation']
        test_index = splits[it]['test']
        
        trainset = dataset[train_index]
        valset = dataset[val_index]
        testset = dataset[test_index]
        
        train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True, drop_last = validate_batch_size(len(trainset), args.batch_size))
        val_loader = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        test_loader = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
              
        cnt = 1
        result_dict = defaultdict(list)
        best_acc_across_folds = -float(np.inf)
        best_loss_across_folds = float(np.inf)
        use_patience_loss = True if args.dataset == "MUTAG" else False
        grid_h =  [64, 32]
        grid_dropout = [0.5, 0]
        n_params = len(grid_dropout) * len(grid_h)
        for dropout in grid_dropout : 
            for hidden_dim in grid_h:   
                params = {"dropout":dropout, "hidden_dim": hidden_dim}

                model = PathNN(
                        features_dim, hidden_dim, args.cutoff, n_classes, dropout, device,
                        residuals = args.residuals, encode_distances=encode_distances
                        ).to(device)

                log.print_and_log(f'Model # Parameters {sum([p.numel() for p in model.parameters()])}')
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                early_stopper = Patience(patience = args.patience, use_loss = use_patience_loss)

                pbar_train = tqdm.tqdm(range(args.epochs), desc = "Epoch 0 Loss 0")
                for epoch in pbar_train:
                    model.train()

                    for idx, data in enumerate(train_loader):
                        data = data.to(device)
                        output, loss = train(model, data,criterion,optimizer)

                    val_loss = 0
                    val_correct = 0 
                    model.eval()
                    for idx, data in enumerate(val_loader):
                        output, loss = test(model, data.to(device),criterion)
                        val_loss += loss.item() * data.num_graphs
                        preds = output.max(1)[1].type_as(data.y)
                        val_correct += torch.sum(preds.eq(data.y.double())).item()


                    val_acc = val_correct/len(val_loader.dataset)
                    val_loss = val_loss / len(val_loader.dataset)

                    if early_stopper.stop(epoch, val_loss, val_acc) : 
                        break

                    best_acc_across_folds = early_stopper.val_acc if  early_stopper.val_acc > best_acc_across_folds else best_acc_across_folds
                    best_loss_across_folds = early_stopper.val_loss if  early_stopper.val_loss < best_loss_across_folds else best_loss_across_folds

                    pbar_train.set_description(f"MS {cnt}/{n_params} Epoch {epoch+1} Val loss {round(val_loss,3)} Val acc {round(val_acc,3)} Best Val Loss {round(early_stopper.val_loss, 3)} Best Val Acc {early_stopper.val_acc:0.3f} Best Fold Val Acc  {best_acc_across_folds:0.3f} Best Fold Val Loss {best_loss_across_folds:0.3f}")
                    
                result_dict['config'].append(params)
                result_dict["best_val_acc"].append(early_stopper.val_acc)
                result_dict["best_val_loss"].append(early_stopper.val_loss)
                log.info(f"MS {cnt}/{n_params} Epoch {epoch+1} Val loss {round(val_loss,3)} Val acc {round(val_acc,3)} Best Val Loss {round(early_stopper.val_loss, 3)} Best Val Acc {early_stopper.val_acc:0.3f} Best Fold Val Acc  {best_acc_across_folds:0.3f} Best Fold Val Loss {best_loss_across_folds:0.3f}")
                    
                cnt+=1

        ################################
        #         MODEL ASSESSMENT    #
        ###############################

        best_i = np.argmin(result_dict["best_val_loss"]) if use_patience_loss else np.argmax(result_dict["best_val_acc"]) 
        best_config = result_dict["config"][best_i]
        log.print_and_log(f"Winner of Model Selection | hidden dim: {best_config['hidden_dim']} | dropout {best_config['dropout']}")
        log.print_and_log(f"Winner Best Val Accuracy {result_dict['best_val_acc'][best_i]:0.5f}")

        test_accs = list() 
        cnt = 1

        model = PathNN(features_dim, best_config['hidden_dim'], args.cutoff, n_classes, best_config["dropout"], device,
                                    use_nn = args.remove_nn, residuals = args.residuals, encode_distances=encode_distances).to(device)
            
        for jj in range(3) :

            model.reset_parameters()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
            save_path = os.path.join('models', args.dataset, 'model_best_'+program_name+'.pth.tar')
            early_stopper = Patience(patience = args.patience, use_loss = use_patience_loss, save_path=save_path)

            pbar_train = tqdm.tqdm(range(args.epochs), desc = "Epoch 0 Loss 0")
                
            for epoch in pbar_train:
                model.train()

                for idx, data in enumerate(train_loader):
                    data = data.to(device)
                    output, loss = train(model, data,criterion,optimizer)

                val_loss = 0
                val_correct = 0 
                model.eval()
                for idx, data in enumerate(val_loader):
                    output, loss = test(model, data.to(device),criterion)
                    val_loss += loss.item() * data.num_graphs
                    preds = output.max(1)[1].type_as(data.y)
                    val_correct += torch.sum(preds.eq(data.y.double())).item()

                val_acc = val_correct/len(val_loader.dataset)
                val_loss = val_loss / len(val_loader.dataset)

                if early_stopper.stop(epoch, val_loss, val_acc, model = model) : 
                    break

                pbar_train.set_description(f"Test {cnt}/3 Epoch {epoch+1} Val loss {round(val_loss,3)} Val acc {round(val_acc,3)} Best Val Loss {round(early_stopper.val_loss, 3)} Best Val Acc {early_stopper.val_acc:0.3f}")
                
            log.info(f"Test {cnt}/3 Epoch {epoch+1} Val loss {round(val_loss,3)} Val acc {round(val_acc,3)} Best Val Loss {round(early_stopper.val_loss, 3)} Best Val Acc {early_stopper.val_acc:0.3f}")
            
            checkpoint = torch.load(save_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            
            test_loss = 0
            test_count = 0
            test_correct = 0
            model.eval()

            for idx, data in enumerate(test_loader):
                output, loss = test(model, data.to(device),criterion)
                test_loss += loss.item() * output.size(0)
                test_count += output.size(0)
                preds = output.max(1)[1].type_as(data.y)
                test_correct += torch.sum(preds.eq(data.y.double())).item()

            test_accs.append((test_correct/test_count))
            cnt+=1
        log.print_and_log(f"Avg test acc {np.mean(test_accs):.3f}, Agv test acc std {np.std(test_accs):.3f}")

        accs_folds.append((test_accs))
        
        log.print_and_log(f'Cross-val iter:{it+1} | Current average test accuracy across folds {np.mean(accs_folds):.5f}')
        log.print_and_log('\n')

        
    accs_folds = np.asarray(accs_folds)
    
    result_dict = {}
    result_dict['test_accuracies'] = accs_folds
    result_dict['best_params'] = best_config
    
    with open(os.path.join('results', args.dataset, 'results_'+program_name+'.pkl'), 'wb') as f :
        pickle.dump(result_dict, f)
    
    log.print_and_log(f"AVERAGE TEST ACC ACROSS FOLDS {np.mean(accs_folds):.5f}")  
    log.print_and_log(f"STD ACROSS FOLDS {np.std(np.mean(accs_folds,axis = 1))}")


if __name__ == '__main__':
    main()
