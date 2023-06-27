import numpy as np
import networkx as nx
import igraph as ig 
import logging
import io 
import json 

import torch
import torch.utils.data as utils

from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_scatter import scatter_add

import tqdm
import torch.nn.functional as F

class PrinterLogger(object) : 
    def __init__(self, logger) : 
        self.logger = logger 
    def print_and_log(self, text) :
        self.logger.info(text)
        print(text)
    def info(self, text) : 
        self.logger.info(text)


class EarlyStopper:

    def stop(self, epoch, val_loss, val_acc=None, test_loss=None, test_acc=None, train_loss=None, train_acc=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return self.train_loss, self.train_acc, self.val_loss, self.val_acc, self.test_loss, self.test_acc, self.best_epoch

class Patience(EarlyStopper):

    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=20, use_loss=True, save_path = None):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1
        self.val_loss, self.val_acc = None, None
        self.save_path = save_path

    def stop(self, epoch, val_loss, val_acc=None, model = None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]) : 
                    torch.save({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                #'optimizer' : optimizer.state_dict(),
                            }, self.save_path)
                return False

            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]) : 
                    torch.save({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                #'optimizer' : optimizer.state_dict(),
                            }, self.save_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

class ModifData(Data) : 
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
            super().__init__(x=x, edge_index = edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        
        if 'index' in key or 'face' in key or "path" in key :
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key :#or "path" in key or "indicator" in key:
            return 1
        else:
            return 0

class PathDataset(Dataset) :
    """
    Computes paths for all nodes in graphs and convert it to pytorch dataset object. 
    """ 
    def __init__(self, Gs, features, y, cutoff, path_type, min_length = 0, undirected = True) : 
        super().__init__()
        self.Gs = Gs
        self.features = features
        self.y = y 
        self.cutoff = cutoff
        self.path_type = path_type 
        self.undirected = undirected
        
        if all([self.path_type is not None, cutoff > 2]) :
            self.gs =  [ig.Graph.from_networkx(g) for g in Gs]            
            self.graph_info = list()
            for g in tqdm.tqdm(self.gs) : 
                self.graph_info.append(generate_paths(g, cutoff, path_type, undirected=undirected))
            self.diameter = max([i[1] for i in self.graph_info])
        else : 
            self.diameter = cutoff
        self.min_length = min_length
        self.datalist = [self._create_data(i) for i in range(self.len())]
        
    def len(self) : 
        return len(self.Gs)
    
    def num_nodes(self) : 
        return sum([G.number_of_nodes() for G in self.Gs])

    def _create_data(self, index) : 
        data = ModifData(**from_networkx(self.Gs[index]).stores[0])
        data.x = torch.FloatTensor(self.features[index])
        data.y = torch.LongTensor([self.y[index]])
        setattr(data, f'path_2', data.edge_index.T.flip(1))
        
        if self.path_type != None : 
            if self.path_type == 'all_simple_paths' : 
                setattr(data, f"sp_dists_2", torch.LongTensor(self.graph_info[index][2][0]).flip(1))
            #setattr(data, f'distances_2', torch.cat([torch.zeros(data.edge_index.size(0), 1), torch.ones(data.edge_index.size(0),1)], dim = 1))
            for jj in range(1, self.cutoff - 1) : 

                paths = torch.LongTensor(self.graph_info[index][0][jj]).view(-1,jj+2)
                if paths.size(0) > 0 : 
                    setattr(data, f'path_{jj+2}', paths.flip(1))
                    if self.path_type == 'all_simple_paths' : 
                        setattr(data, f"sp_dists_{jj+2}", torch.LongTensor(self.graph_info[index][2][jj]).flip(1))
                else : 
                    setattr(data, f'path_{jj+2}', torch.empty(0,jj+2).long())
                    
                    if self.path_type == 'all_simple_paths' : 
                        setattr(data, f"sp_dists_{jj+2}", torch.empty(0,jj+2).long())
        return data 

    def get(self, index) : 
        return self.datalist[index]


def validate_batch_size(length, batch_size) : 
    """Returns True if the last batch has size = 1"""
    if length % batch_size == 1 : 
        return True
    return False
        
def load_data(ds_name, use_node_labels, use_node_attributes, degree_as_tag = False):    
    """Read a text file containing adjacency matrix and node initial representations and converts it to 
    a list of networkx graph objects."""

    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(ds_name,ds_name), dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    xx = []
    if use_node_labels:
        x = np.loadtxt("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), dtype=np.int64).reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
        xx.append(x)
    if use_node_attributes:
        x = np.loadtxt("datasets/%s/%s_node_attributes.txt"%(ds_name,ds_name), dtype=np.float64, delimiter=',')
        xx.append(x)
    if degree_as_tag : 
        x = A.sum(axis=1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
        xx.append(x)
    elif not use_node_labels and not use_node_attributes and not degree_as_tag:
        x = np.ones((A.shape[0], 1))
        xx.append(x)
        
    x = np.hstack(xx)
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx+graph_size[i],idx:idx+graph_size[i]])
        features.append(x[idx:idx+graph_size[i],:])
        idx += graph_size[i]

    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), dtype=np.int64)

    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    y = np.array([class_labels[i] for i in range(class_labels.size)])
    
    Gs = list()
    for i in range(len(adj)):
        Gs.append(nx.from_scipy_sparse_array(adj[i]))

    with open(f"datasets/{ds_name}/{ds_name}_splits.json", "r") as f : 
        splits = json.load(f)
    return Gs, features, y, splits


def generate_paths(g, cutoff, path_type, weights = None, undirected = True) : 
    """
    Generates paths for all nodes in the graph, based on specified path type. This function uses igraph rather than networkx
    to generate paths as it gives a more than 10x speedup. 
    """
    if undirected and g.is_directed() : 
        g.to_undirected()

    path_length = np.array(g.distances())
    if path_type != "all_simple_paths" : 
        diameter = g.diameter(directed = False) 
        diameter = diameter+1 if diameter+1 < cutoff else cutoff

    else : 
        diameter = cutoff

    X = [[] for i in range(cutoff-1)] 
    sp_dists = [[] for i in range(cutoff-1)] 

    for n1 in range(g.vcount()) : 

        if path_type == "all_simple_paths" : 
            paths_ = g.get_all_simple_paths(n1, cutoff = cutoff-1)
                
            for path in paths_: 
                idx = len(path)-2
                if len(path) > 0 : 
                    X[idx].append(path)
                    # Adding geodesic distance 
                    sp_dist = []
                    for node in path : 
                        sp_dist.append(path_length[n1, node])
                    sp_dists[idx].append(sp_dist)
                        
        else : 
            valid_ngb = [i for i in np.where((path_length[n1] <= cutoff - 1) & (path_length[n1] > 0))[0] if i > n1]
            for n2 in valid_ngb : 
                if path_type == "shortest_path" :
                    paths_ = g.get_shortest_paths(n1,n2, weights=weights)
                elif path_type == "all_shortest_paths" : 
                    paths_ = g.get_all_shortest_paths(n1, n2, weights=weights) 

                for path in paths_ : 
                    idx = len(path)-2
                    X[idx].append(path)
                    X[idx].append(list(reversed(path)))

    return X, diameter, sp_dists



