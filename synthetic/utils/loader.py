
from torch_geometric.loader import DataLoader
from torch import LongTensor
def prepare_loaders(dataset, splits, batch_size = 32, index = 0) : 

    train_idx = LongTensor(splits["train"][index])
    val_idx = LongTensor(splits["val"][index])
    test_idx = LongTensor(splits["test"][index])

    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size)
    test_loader = DataLoader(dataset[test_idx], batch_size=batch_size)

    return train_loader, val_loader, test_loader
