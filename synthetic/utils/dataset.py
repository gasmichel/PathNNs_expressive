import numpy as np
import igraph as ig 

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils.convert import from_networkx, to_networkx
import tqdm


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
    def __init__(self, Gs, features, y, cutoff, path_type, min_length = 0, undirected = True, task = 'classification') : 
        super().__init__()
        self.Gs = Gs
        self.features = features
        self.y = y 
        self.cutoff = cutoff
        self.path_type = path_type 
        self.undirected = undirected
        self.task = task 

        if all([self.path_type is not None, cutoff >= 2]) :
            self.gs =  [ig.Graph.from_networkx(g) for g in Gs]            
            self.graph_info = list()
            for g in tqdm.tqdm(self.gs) : 
                self.graph_info.append(fast_generate_paths2(g, cutoff, path_type, undirected=undirected))
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
        if self.task == 'iso' : 
            x_class = torch.DoubleTensor
        else : 
            x_class = torch.FloatTensor
        data = ModifData(**from_networkx(self.Gs[index]).stores[0])
        data.x = x_class(self.features[index])
        data.y = torch.LongTensor([self.y[index]])

        setattr(data, f'path_2', data.edge_index.T.flip(1))
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

       
class PathTransform(object):
    def __init__(self, path_type, cutoff, r = None):
        self.cutoff = cutoff
        self.r = r 
        self.path_type = path_type 

    def __call__(self, data):
        G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
        setattr(data, f'path_2', data.edge_index.T)
        
        graph_info = fast_generate_paths2(G, self.cutoff, self.path_type, undirected=True)
        degree = torch.LongTensor(G.degree())    
        if self.r : 
            sampled_paths = [sample(torch.LongTensor(graph_info[0][sample_k-2]).view(-1,sample_k), r = self.r)  for sample_k in range(3, self.cutoff+1)]
            
        for jj in range(1, self.cutoff - 1) : 
            if all([self.r,jj>0]) : 
                paths = torch.LongTensor(sampled_paths[jj-1])
            else : 
                paths = torch.LongTensor(graph_info[0][jj]).view(-1,jj+2)
            setattr(data, f'path_{jj+2}', paths)
        data.max_cutoff = [i+2 for i in range(self.cutoff-1) if getattr(data, f"path_{i+2}").size(0) > 0][-1]

        return ModifData(**data.stores[0])


def fast_generate_paths2(g, cutoff, path_type, weights = None, undirected = True) : 
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


