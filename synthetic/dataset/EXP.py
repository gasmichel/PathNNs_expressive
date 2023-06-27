import os 
import numpy as np
import networkx as nx
import torch
from utils.dataset import PathDataset
from torch_geometric.data import Data
from collections import defaultdict

class EXP(torch.utils.data.Dataset):
    def __init__(self, root = "../data/EXP/", task = "iso", cutoff = 5, path_type = "shortest_path", n_splits = 4):
        self.cutoff = cutoff
        self.path_type = path_type
        self.raw_file_name = os.path.join(root, "EXP.txt" if task == "iso" else "CEXP.txt")
        self.task = task
        self.n_splits = n_splits
        self._prepare()

    def _prepare(self):

        Gs = []
        ys = []
        features =[]
        with open(self.raw_file_name, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for i in range(num_graphs):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex))
                features.append(np.ones((num_vertex, 1)))
                ys.append(float(graph_meta[1]))
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    for k in range(2,len(vertex)):
                        curr_graph[j, int(vertex[k])] = np.ones((1))
                Gs.append(nx.from_numpy_array(curr_graph))
        self.dataset = PathDataset(Gs, features, ys, self.cutoff, self.path_type, task=self.task)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_all_splits_idx(self) : 
        splits = defaultdict(list)
        val_size = int(self.__len__() * 0.1)
        test_size = int(self.__len__() * 0.15)
        for it in range(self.n_splits) : 
            indices = np.arange(self.__len__()) 
            val_idx = np.arange(start = (it) * val_size, stop = (it+1)*val_size)
            test_idx = np.arange(start = (it) * test_size, stop = (it+1)*test_size) 
            splits["val"].append(indices[val_idx])
            remaining_indices = np.delete(indices, val_idx)
            splits["test"].append(remaining_indices[test_idx])
            remaining_indices = np.delete(remaining_indices, test_idx)
            splits["train"].append(remaining_indices)
        return splits

if __name__ == "__main__" : 
    dataset = EXP()
    print(dataset[0])
    dataset = EXP(task = "class")
    print(dataset[0])
    splits = dataset.get_all_splits_idx()
    print(splits)