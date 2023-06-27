import networkx as nx
import torch
from utils.dataset import to_networkx, fast_generate_paths2, ModifData
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import tqdm 
import igraph as ig 

class SR25(torch.utils.data.Dataset):
    """
        Circular Skip Link Graphs: 
        Source: https://github.com/PurdueMINDS/RelationalPooling/
    """
    
    def __init__(self, root="data/SR25/", dataset_name = "sr16622.g6", cutoff = 5, path_type = "all_simple_paths"):
        self.name = "SR25"
        self.cutoff = cutoff
        self.root = root + dataset_name
        self.path_type = path_type
        self._prepare()

    def _create_data(self, data) : 
        G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
        setattr(data, f'path_2', data.edge_index.T.flip(1))
        
        graph_info = fast_generate_paths2(G, self.cutoff, self.path_type, undirected=True)
        self.graph_info = graph_info
        if self.path_type == 'all_simple_paths' : 
            setattr(data, f"sp_dists_2", torch.LongTensor(graph_info[2][0]).flip(1))
        for jj in range(1, self.cutoff - 1) : 

            paths = torch.LongTensor(graph_info[0][jj]).view(-1,jj+2)
            distances = torch.LongTensor(graph_info[2][jj])
            
            setattr(data, f'path_{jj+2}', paths.flip(1))
            if self.path_type == 'all_simple_paths' : 
                setattr(data, f"sp_dists_{jj+2}", distances.flip(1))
        max_cutoff = [i+2 for i in range(self.cutoff-1) if getattr(data, f"path_{i+2}").size(0) > 0][-1]

        return ModifData(**data.stores[0]), max_cutoff
        
    def _prepare(self):
        dataset = nx.read_graph6(self.root)

        self.data_list, cutoffs = [], []
        for i,datum in enumerate(tqdm.tqdm(dataset)):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data = Data(edge_index=edge_index, x=x, y=0)
            data, max_cutoff = self._create_data(data)
            data.graph_indicator = i 
            self.data_list.append(data)
            cutoffs.append(max_cutoff)
        self.max_diameter = max(cutoffs)
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == "__main__" : 
    dataset = SR25()
    print(dataset[0])



        