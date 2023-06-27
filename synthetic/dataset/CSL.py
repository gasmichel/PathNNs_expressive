import numpy as np
import networkx as nx
import time
import torch
from torch_geometric.utils.convert import from_networkx, to_networkx
import pickle, os 
from utils.dataset import PathDataset
import csv 

class CSL(torch.utils.data.Dataset):
    """
        Circular Skip Link Graphs: 
        Source: https://github.com/PurdueMINDS/RelationalPooling/
    """
    
    def __init__(self, root="../data/CSL/", cutoff = 5, path_type = "shortest_path"):
        self.name = "CSL"
        self.cutoff = cutoff
        self.path_type = path_type
        self.adj_list = pickle.load(open(os.path.join(root, 'graphs_Kary_Deterministic_Graphs.pkl'), 'rb'))
        self.graph_labels = torch.load(os.path.join(root, 'y_Kary_Deterministic_Graphs.pt'))
        self.graph_lists = []
        self.n_classes = len(torch.unique(self.graph_labels))
        self.n_samples = len(self.graph_labels)
        self.num_node_type = 1 #41
        self.num_edge_type = 1 #164
        self.degrees = []
        self.n_splits = 5
        self.root_dir = root
        self._prepare()
        
    def _prepare(self):
        t0 = time.time()
        graph_list = []
        feature_list = []
        print("[I] Preparing Circular Skip Link Graphs v4 ...")
        for idx in range(self.n_samples):
            G = nx.from_scipy_sparse_array(self.adj_list[idx])
            G.remove_edges_from(nx.selfloop_edges(G))
            X = np.ones((len(G), 1))
            degree = nx.degree(G)
            graph_list.append(G)
            feature_list.append(X)
            self.degrees.append(degree)
        self.dataset = PathDataset(graph_list, feature_list, self.graph_labels.numpy(), self.cutoff, self.path_type )
        print("[I] Finished preparation after {:.4f}s".format(time.time()-t0))
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_all_splits_idx(self):
        """
            - Split total number of graphs into 3 (train, val and test) in 3:1:1
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 5 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 5 fold have unique test set.
        """

        all_idx = {}
            
        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(self.root_dir + self.name + '_'+ section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

if __name__ == "__main__" : 
    dataset = CSL()
    print(dataset[0])
    print(dataset.get_all_splits_idx())