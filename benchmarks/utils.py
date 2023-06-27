import numpy as np
import networkx as nx
import igraph as ig
import logging
import io
import json

import torch
import torch.utils.data as utils

from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_scatter import scatter_add
from sklearn.metrics import roc_auc_score, average_precision_score

import tqdm
import torch.nn.functional as F


def eval_ap(y_true, y_pred):
    """
    compute Average Precision (AP) averaged across tasks
    """

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute Average Precision."
        )

    return {"ap": sum(ap_list) / len(ap_list)}


class PrinterLogger(object):
    def __init__(self, logger):
        self.logger = logger

    def print_and_log(self, text):
        self.logger.info(text)
        print(text)

    def info(self, text):
        self.logger.info(text)


class EarlyStopper:
    def stop(
        self,
        epoch,
        val_loss,
        val_acc=None,
        test_loss=None,
        test_acc=None,
        train_loss=None,
        train_acc=None,
    ):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return (
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.test_loss,
            self.test_acc,
            self.best_epoch,
        )


class Patience(EarlyStopper):

    """
    Implement common "patience" technique
    """

    def __init__(self, patience=20, use_loss=True, save_path=None, maximize=True):
        if use_loss or not maximize:
            self.local_val_optimum = float("inf")
            self.val_acc = float("inf")

        else:
            self.local_val_optimum = -float("inf")
            self.val_acc = -float("inf")

        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1
        self.val_loss = None
        self.save_path = save_path
        self.maximize = maximize

    def stop(self, epoch, val_loss, val_acc=None, model=None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]):
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            #'optimizer' : optimizer.state_dict(),
                        },
                        self.save_path,
                    )
                return False

            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if self.maximize:
                cond = val_acc >= self.local_val_optimum
            else:
                cond = val_acc <= self.local_val_optimum
            if cond:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]):
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            #'optimizer' : optimizer.state_dict(),
                        },
                        self.save_path,
                    )
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience


class ModifData(Data):
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
        super().__init__(x=x, edge_index=edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):

        if "index" in key or "path" in key:
            return self.num_nodes
        elif "indices" in key:
            return self.num_edges
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "index" in key or "face" in key:
            return 1
        else:
            return 0


class RandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    This sampler saves the random permutation applied to the training data,
    so it is available for further use (e.g. for saving).
    The permutation is saved in the 'permutation' attribute.
    The DataLoader can now be instantiated as follows:
    >>> data = Dataset()
    >>> dataloader = DataLoader(dataset=data, batch_size=32, shuffle=False, sampler=RandomSampler(data))
    >>> for batch in dataloader:
    >>>     print(batch)
    >>> print(dataloader.sampler.permutation)
    For convenience, one can create a method in the dataloader class to access the random permutation directly, e.g:
    class MyDataLoader(DataLoader):
        ...
        def get_permutation(self):
            return self.sampler.permutation
        ...
    """

    def __init__(self, data_source, num_samples=None, replacement=False):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples)
        self.permutation = None

    def __iter__(self):
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)


def get_loader(dataset, batch_size=1, shuffle=True, drop_last=False):

    sampler = RandomSampler(dataset) if shuffle is True else None

    # 'shuffle' needs to be set to False when instantiating the DataLoader,
    # because pytorch  does not allow to use a custom sampler with shuffle=True.
    # Since our shuffler is a random shuffler, either one wants to do shuffling
    # (in which case he should instantiate the sampler and set shuffle=False in the
    # DataLoader) or he does not (in which case he should set sampler=None
    # and shuffle=False when instantiating the DataLoader)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        shuffle=False,  # if shuffle is not None, must stay false, ow is shuffle is false
        pin_memory=True,
    )


def validate_batch_size(length, batch_size):
    """Returns True if the last batch has size = 1"""
    if length % batch_size == 1:
        return True
    return False


def fast_generate_paths2(g, cutoff, path_type, weights=None, undirected=True, r=None):
    if undirected and g.is_directed():
        g.to_undirected()

    path_length = np.array(g.distances())
    if path_type != "all_simple_paths":
        diameter = g.diameter(directed=False)
        diameter = diameter + 1 if diameter + 1 < cutoff else cutoff

    else:
        diameter = cutoff

    X = [[] for i in range(cutoff - 1)]
    sp_dists = [[] for i in range(cutoff - 1)]

    for n1 in range(g.vcount()):
        if path_type == "all_simple_paths":
            paths_ = g.get_all_simple_paths(n1, cutoff=cutoff - 1)

            for path in paths_:
                # if len(path) >= min_length and len(path) <= cutoff :
                idx = len(path) - 2
                if len(path) > 0:
                    X[idx].append(path)
                    sp_dist = []
                    for node in path:
                        sp_dist.append(path_length[n1, node])
                    sp_dists[idx].append(sp_dist)

        else:
            valid_ngb = [
                i
                for i in np.where(
                    (path_length[n1] <= cutoff - 1) & (path_length[n1] > 0)
                )[0]
                if i > n1
            ]
            for n2 in valid_ngb:
                if path_type == "shortest_path":
                    paths_ = g.get_shortest_paths(n1, n2, weights=weights)
                elif path_type == "all_shortest_paths":
                    paths_ = g.get_all_shortest_paths(n1, n2, weights=weights)

                for path in paths_:
                    idx = len(path) - 2
                    X[idx].append(path)
                    X[idx].append(list(reversed(path)))

    return X, diameter, sp_dists
