from utils import * 
from evaluator import Evaluator
from model_ogb import EdgePathNN

import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim

from tqdm import tqdm
import argparse
import numpy as np
import datetime 
import pickle 
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset
from preprocess_data import get_dataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.L1Loss()
mse_reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()
    loss_curve = list()
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            elif task_type == 'regression':
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size()))
            elif task_type == 'mse_regression' : 
                loss = mse_reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size()))
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.detach().cpu().item())
    return loss_curve

@torch.no_grad()
def eval(model, device, loader, evaluator, task_type):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            elif task_type == 'regression':
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size()))
            elif task_type == 'mse_regression' : 
                loss = mse_reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled].view(pred.size()))
                
            total_loss += loss.item() * batch.num_graphs
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), total_loss / len(loader.dataset)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--path-type', default='shortest_path', 
                        help='Type of extracted path')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--patience', default=50, type = int, help='max early stopping round')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--residuals', default = False, action = 'store_true')
    parser.add_argument('--use_edge_attr', default = False, action = 'store_true')
    parser.add_argument('--config_run', type=str, help = 'Config file (json) that contains setup for the test run.')
    parser.add_argument('--weight_decay', default = 0, type=float, help = 'weight_decay on optimizer.')
    parser.add_argument('--output_dir', default = './', type=str)

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ## Reproductible results
    torch.manual_seed(2022)
    np.random.seed(2022)

    #Return to benchmarks directory in case we start script for script directory
    if os.getcwd()[-10:] != 'benchmarks' : 
        os.chdir(os.path.dirname(os.getcwd()) )

    ###### LOGGING AND DIRECTORY ######
    if not os.path.exists(os.path.join(os.getcwd(), 'results', args.dataset)) : 
        os.makedirs(os.path.join(os.getcwd(), 'results', args.dataset))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs', args.dataset)) : 
        os.makedirs(os.path.join(os.getcwd(), 'logs', args.dataset))
    if not os.path.exists(os.path.join('models', args.dataset)) : 
        os.makedirs(os.path.join('models', args.dataset))
    now = "_" + "-".join(str(datetime.datetime.today()).split()).split('.')[0].replace(':','.')
    edge_flag = "_edgea" if args.use_edge_attr else ""
    weight_decay = "wd_"+str(args.weight_decay) if args.weight_decay > 0 else ""
    program_name = str(args.path_type)+f'_bs_{args.batch_size}'+'_lr_'+str(args.lr)+weight_decay+edge_flag+now

    logging.basicConfig(filename=os.path.join('logs', args.dataset, program_name+'.log'), level=logging.INFO, filemode="w")
    log = PrinterLogger(logging.getLogger(__name__) )
    
    encode_distances = True if args.path_type == 'all_simple_paths' else False

    #Opening config file
    with open(args.config_run, 'r') as f : 
        params = json.load(f)

    log.print_and_log(params)
    experiment_name = str(params['hidden_dim']) +'_drop'+ str(params['dropout']) + '_' + \
                      str(params['cutoff']) + '_' + params['readout'] + '_' + params['path_agg']
    model_save_path = os.path.join('models', args.dataset, 'model_best' +program_name+experiment_name+'.pth.tar')
    test_perfs_at_best, test_perfs_at_end = [], []
    val_perfs_at_best, val_perfs_at_end = [], []
    train_perfs_at_end, train_losses = [], []
    best_epochs = []

    dataset = get_dataset(args.dataset, params['cutoff'], args.path_type, args.output_dir)
    
    if args.dataset in ["ogbg-molhiv", "peptides-functional"] : 
        maximize = True
    else : 
        maximize = False
    
    if "peptides" in args.dataset : 
        split_val_name = 'val'
    else : 
        split_val_name = "valid"
    if args.dataset == 'ZINC' : 
        num_embeddings_v, num_embeddings_e = dataset.num_node_type, dataset.num_edge_type
    else : 
        num_embeddings_v, num_embeddings_e = None, None

    evaluator = Evaluator(dataset.num_tasks, dataset.eval_metric)

    split_idx = dataset.get_idx_split()
    
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx[split_val_name]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)


    if 'n_seeds' in params.keys() : 
        n_seeds = params['n_seeds']
    else : 
        # Default to 4 test seeds
        n_seeds = 4
    log.print_and_log("\n" + "-"*15 + f" TESTING OVER {n_seeds} SEEDS" + "-"*15+ "\n")

    model = EdgePathNN(params['hidden_dim'], params['cutoff'], dataset.num_tasks,
                            device, residuals = args.residuals, use_edge_attr=args.use_edge_attr,
                            num_embeddings_v = num_embeddings_v, num_embeddings_e = num_embeddings_e, 
                            encode_distances=encode_distances,
                            readout = params['readout'], path_agg= params['path_agg'],
                            dropout=params['dropout']).to(device) 

    log.print_and_log(f'Model # Parameters {sum([p.numel() for p in model.parameters()])}')
    for seed in range(n_seeds) : 
        model.reset_parameters()
        if args.weight_decay == 0 : 
            optim_class = optim.Adam
        else : 
            optim_class = optim.AdamW
            
        optimizer = optim_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.dataset in ["ZINC", "peptides-functional", "peptides-structural"] : 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, mode = "min")
        else : 
            scheduler = None
            
        early_stopper = Patience(patience = args.patience, use_loss = False, save_path=model_save_path, maximize=maximize)

        pbar = tqdm(range(args.epochs), desc = f"{seed+1}/{n_seeds}")
        for epoch in pbar:
            train_loss_curve = train(model, device, train_loader, optimizer, dataset.task_type)
            train_losses.extend(train_loss_curve)
            valid_perf, val_loss = eval(model, device, valid_loader, evaluator, dataset.task_type)
            if maximize : 
                if valid_perf[dataset.eval_metric] >= early_stopper.val_acc : 
                    test_perf_at_best, test_loss_at_best = eval(model, device, test_loader, evaluator, dataset.task_type)
            else : 
                if valid_perf[dataset.eval_metric] <= early_stopper.val_acc : 
                    test_perf_at_best, test_loss_at_best = eval(model, device, test_loader, evaluator, dataset.task_type)

            if scheduler is not None : 
                scheduler.step(valid_perf[dataset.eval_metric])
                if args.dataset == 'ZINC' and optimizer.param_groups[0]['lr'] <= 1e-5 : 
                    break 
            if early_stopper.stop(epoch, val_loss, valid_perf[dataset.eval_metric], model = model) : 
                break

            pbar.set_description(f"{seed+1}/{n_seeds}  Epoch {epoch+1} Val loss {round(val_loss,3)} Val perf {valid_perf[dataset.eval_metric]:.3f} Best Val Loss {early_stopper.val_loss:0.3f} Best Val Perf {early_stopper.val_acc:0.3f} Test Perf @ Best {test_perf_at_best[dataset.eval_metric]:0.3f}")
        
        test_perf_at_end, test_loss_at_end = eval(model, device, test_loader, evaluator, dataset.task_type)
        train_perf_at_end, train_loss_at_end = eval(model, device, train_loader, evaluator, dataset.task_type)



        train_perfs_at_end.append(train_perf_at_end[dataset.eval_metric])
        val_perfs_at_end.append(valid_perf[dataset.eval_metric])
        test_perfs_at_end.append(test_perf_at_end[dataset.eval_metric])
        val_perfs_at_best.append(early_stopper.val_acc)
        test_perfs_at_best.append(test_perf_at_best[dataset.eval_metric])
        best_epochs.append(early_stopper.best_epoch+1)

        msg = (
        f'============= Results {seed+1}/{n_seeds}=============\n'
        f'Dataset:        {args.dataset}\n'
        f'-------------  Best epoch ------------------\n'
        f'Best epoch:     {early_stopper.best_epoch+1}\n'
        f'Validation:     {early_stopper.val_acc:0.4f}    Seed Average: {np.mean(val_perfs_at_best):0.4f} +/- {np.std(val_perfs_at_best):0.4f}\n'
        f'Test:           {test_perf_at_best[dataset.eval_metric]:0.4f}    Seed Average: {np.mean(test_perfs_at_best):0.4f} +/- {np.std(test_perfs_at_best):0.4f}\n'
        '--------------- Last epoch -----------------\n'
        f'Train:          {train_perf_at_end[dataset.eval_metric]:0.4f}    Seed Average: {np.mean(train_perfs_at_end):0.4f} +/- {np.std(train_perfs_at_end):0.4f}\n'
        f'Validation:     {valid_perf[dataset.eval_metric]:0.4f}    Seed Average: {np.mean(val_perfs_at_end):0.4f} +/- {np.std(val_perfs_at_end):0.4f}\n'
        f'Test:           {test_perf_at_end[dataset.eval_metric]:0.4f}    Seed Average: {np.mean(test_perfs_at_end):0.4f} +/- {np.std(test_perfs_at_end):0.4f}\n'
       '-------------------------------------------\n\n')
        
        log.print_and_log(msg)

    results = {
        "arch" : params, 
        "perfs_at_end" : {"train" : np.asarray(train_perfs_at_end), "val" : np.asarray(val_perfs_at_end), "test" : np.asarray(test_perfs_at_end)},
        "perfs_at_best" : {"val" : np.asarray(val_perfs_at_best), "test" : np.asarray(test_perfs_at_best)},
        "best_epochs" : np.asarray(best_epochs),
        "train_loss_curve" : np.asarray(train_losses) } 

    with open(os.path.join("results", "results_" + program_name + ".pkl"), "wb") as f :
        pickle.dump(results, f)


if __name__ == "__main__":
    main()