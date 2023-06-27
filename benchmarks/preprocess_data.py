from utils import *

import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from tqdm import tqdm
import argparse
import time
import numpy as np
import networkx as nx

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.datasets import ZINC
from torch_geometric.data import InMemoryDataset
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url
import pickle
import hashlib
import os.path as osp
import pickle
import shutil
import pandas as pd
from collections import defaultdict
from ogb.utils import smiles2graph


class PathTransform(object):
    def __init__(self, path_type, cutoff):
        self.cutoff = cutoff
        self.path_type = path_type

    def __call__(self, data):
        if self.path_type is None and self.cutoff is None:
            data.pp_time = 0
            return data

        setattr(data, f"path_2", data.edge_index.T.flip(1))
        setattr(
            data,
            f"edge_indices_2",
            get_edge_indices(
                data.x.size(0), data.edge_index, data.edge_index.T.flip(1)
            ),
        )

        if self.cutoff == 2 and self.path_type is None:
            data.pp_time = 0
            return ModifData(**data.stores[0])

        t0 = time.time()
        G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
        if self.path_type == "all_simple_paths":
            setattr(
                data,
                f"sp_dists_2",
                torch.cat(
                    [torch.ones(data.num_edges, 1), torch.zeros(data.num_edges, 1)],
                    dim=1,
                ).long(),
            )
        graph_info = fast_generate_paths2(
            G, self.cutoff, self.path_type, undirected=True
        )

        cnt = 0
        for jj in range(1, self.cutoff - 1):
            paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
            setattr(data, f"path_{jj+2}", paths.flip(1))
            setattr(
                data,
                f"edge_indices_{jj+2}",
                get_edge_indices(data.x.size(0), data.edge_index, paths.flip(1)),
            )
            if self.path_type == "all_simple_paths":
                if len(paths) > 0:
                    setattr(
                        data,
                        f"sp_dists_{jj+2}",
                        torch.Tensor(graph_info[2][jj]).long().flip(1),
                    )
                else:
                    setattr(data, f"sp_dists_{jj+2}", torch.empty(0, jj + 2).long())
                    cnt += 1
        data.max_cutoff = self.cutoff
        data.cnt = cnt
        data.pp_time = time.time() - t0
        return ModifData(**data.stores[0])


def get_edge_indices(size, edge_index_n, paths):
    index_tensor = torch.zeros(size, size, dtype=torch.long, device=paths.device)
    index_tensor[edge_index_n[0], edge_index_n[1]] = torch.arange(
        edge_index_n.size(1), dtype=torch.long, device=paths.device
    )
    indices = []
    for i in range(paths.size(1) - 1):
        indices.append(index_tensor[paths[:, i], paths[:, i + 1]].unsqueeze(1))

    return torch.cat(indices, -1)


class ZincDataset(InMemoryDataset):
    """This is ZINC from the Benchmarking GNNs paper. This is a graph regression task."""

    def __init__(
        self,
        root,
        path_type="shortest_path",
        cutoff=3,
        transform=None,
        pre_filter=None,
        pre_transform=None,
        subset=True,
        n_jobs=2,
    ):
        self.name = "ZINC"
        self._subset = subset
        self._n_jobs = n_jobs
        self.path_type = path_type
        self.cutoff = cutoff
        self.task_type = "regression"
        self.num_node_type = 28
        self.num_edge_type = 4
        self.num_tasks = 1
        self.eval_metric = "mae"
        super(ZincDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices, idx = self.load_dataset()
        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        name = self.name
        return [f"{name}.pt", f"{name}_idx.pt"]

    def download(self):
        # Instantiating this will download and process the graph dataset.
        ZINC(self.raw_dir, subset=self._subset)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        train_data = ZINC(self.raw_dir, subset=self._subset, split="train")
        val_data = ZINC(self.raw_dir, subset=self._subset, split="val")
        test_data = ZINC(self.raw_dir, subset=self._subset, split="test")

        data_list = []
        idx = []
        start = 0
        t0 = time.time()
        train_data = [self.convert(data) for data in train_data]
        data_list += train_data
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        val_data = [self.convert(data) for data in val_data]
        data_list += val_data
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        test_data = [self.convert(data) for data in test_data]
        data_list += test_data
        idx.append(list(range(start, len(data_list))))

        self.preprocessing_time = time.time() - t0
        path = self.processed_paths[0]
        print(f"Saving processed dataset in {path}....")
        torch.save(self.collate(data_list), path)

        path = self.processed_paths[1]
        print(f"Saving idx in {path}....")
        torch.save(idx, path)

    def convert(self, data):

        if self.path_type is None and self.cutoff is None:
            return data

        data.x = data.x.squeeze(1)
        setattr(data, f"path_2", data.edge_index.T.flip(1))
        setattr(
            data,
            f"edge_indices_2",
            get_edge_indices(
                data.x.size(0), data.edge_index, data.edge_index.T.flip(1)
            ),
        )

        if self.cutoff == 2 and self.path_type is None:
            return ModifData(**data.stores[0])

        else:
            G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
            if self.path_type == "all_simple_paths":
                setattr(
                    data,
                    f"sp_dists_2",
                    torch.cat(
                        [torch.ones(data.num_edges, 1), torch.zeros(data.num_edges, 1)],
                        dim=1,
                    ).long(),
                )
            graph_info = fast_generate_paths2(
                G, self.cutoff, self.path_type, undirected=True
            )

            cnt = 0
            for jj in range(1, self.cutoff - 1):
                paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
                setattr(data, f"path_{jj+2}", paths.flip(1))
                setattr(
                    data,
                    f"edge_indices_{jj+2}",
                    get_edge_indices(data.x.size(0), data.edge_index, paths.flip(1)),
                )
                if self.path_type == "all_simple_paths":
                    if len(paths) > 0:
                        setattr(
                            data,
                            f"sp_dists_{jj+2}",
                            torch.LongTensor(graph_info[2][jj]).flip(1),
                        )
                    else:
                        setattr(data, f"sp_dists_{jj+2}", torch.empty(0, jj + 2).long())
                        cnt += 1

            data.max_cutoff = self.cutoff
            data.cnt = cnt
            return ModifData(**data.stores[0])

    def get_idx_split(self):
        return {"train": self.train_ids, "valid": self.val_ids, "test": self.test_ids}


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "peptides-functional")
        self.task_type = "classification"
        self.num_tasks = 10
        self.eval_metric = "ap"
        self.root = root

        self.url = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
        self.version = (
            "701eb743e899f4d793f0e13c8fa5a1b4"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.preprocessing_time = sum(self.data.pp_time).item()

    @property
    def raw_file_names(self):
        return "peptide_multi_class_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_multi_class_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([eval(data_df["labels"].iloc[i])])

            data_list.append(data)

        t0 = time.time()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root, "splits_random_stratified_peptide.pickle")
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


class PeptidesStructuralDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        PyG dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.
        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.
        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.task_type = "regression"
        self.num_tasks = 11
        self.eval_metric = "mae"
        self.folder = osp.join(root, "peptides-structural")

        self.url = "https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1"
        self.version = (
            "9786061a34298a0684150f2e4ff13f47"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "peptide_structure_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_structure_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]
        target_names = [
            "Inertia_mass_a",
            "Inertia_mass_b",
            "Inertia_mass_c",
            "Inertia_valence_a",
            "Inertia_valence_b",
            "Inertia_valence_c",
            "length_a",
            "length_b",
            "length_c",
            "Spherocity",
            "Plane_best_fit",
        ]
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.
        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(
            self.root, "splits_random_stratified_peptide_structure.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


def get_dataset(dataset, cutoff, path_type, output_dir="./"):
    print(f"Preprocessing {dataset} - {path_type}".upper())
    if not os.path.exists(os.path.join(output_dir, "dataset")):
        os.makedirs(os.path.join(output_dir, "dataset"))
    root = os.path.join(
        output_dir, "dataset", dataset + "_" + str(path_type) + "_cutoff_" + str(cutoff)
    )

    if dataset in ["ogbg-molhiv", "ogbg-molpcba"]:
        data = PygGraphPropPredDataset(
            name=dataset, pre_transform=PathTransform(path_type, cutoff), root=root
        )
        data.preprocessing_time = sum([i.pp_time for i in data]).item()

    elif dataset == "ZINC":
        data = ZincDataset(root=root, path_type=path_type, cutoff=cutoff)
    elif dataset == "peptides-functional":
        data = PeptidesFunctionalDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "peptides-structural":
        data = PeptidesStructuralDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--min_cutoff", type=int, default=3)
    parser.add_argument(
        "--max_cutoff", type=int, default=5, help="Max length of shortest paths"
    )
    args = parser.parse_args()

    for cutoff in range(args.min_cutoff, args.max_cutoff + 1):
        # for dataset in ["ogbg-molhiv", "ogbg-molpcba", "ZINC", "peptides-functional", "peptides-structural"] :
        for dataset in [args.dataset]:

            for path_type in [
                "shortest_path",
                "all_shortest_paths",
                "all_simple_paths",
            ]:

                data = get_dataset(dataset, cutoff, path_type)
                print(data.preprocessing_time)
