# Path Neural Networks: Expressive and Accurate Graph Neural Networks

This repository contains scripts to reproduce experiments presented in the paper **Gaspard Michel, Giannis Nikolentzos, Johannes Lutzeyer, Michalis Vazirgiannis: *Path Neural Networks: Expressive and Accurate Graph Neural Networks*** (ICML 2023).

![Alt text](figures/pathnn_k1.png?raw=true)
![Alt text](figures/pathnn_k2.png?raw=true)

## Installation

----`

The following were used for this project:

- `python 3.9x`
- `PyTorch 1.12.1`
- `Cuda 10.2`
- `torch_geometric 2.2.0`

We recommend to follow these steps to create a virtual environment before running any expriments:

Create the virtual environment:

```
conda create --name pathnn python=3.9 
conda activate pathnn
conda install pip
```

Install dependencies:

```
#PyTorch install
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

#PyG install
pip install torch-scatter==2.1.0 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
pip install torch-geometric==2.2.0

#Other dependencies
pip install -r requirements.txt
```

## Run Experiments

----

Scripts for each experiment done in the paper are provided, in each `experiment_name/scripts/` subfolders.
 `synthetic` folder contains synthetic experiments described in Section 5.1 of the paper, `TUDatasets` folder contains experiments on the TUDatasets and `benchmarks` folder contains experiments on ZINC12K, Peptides-functional, Peptides-structural and ogbg-molhiv.
These experiments are described in Section 5.2 of the paper.

To run an experiment, open a terminal window, navigate to the desired `scripts` subfolder, and execute one of the bash file with the following command:

```
sh <dataset_name>_<path_type>.sh
```

where `<dataset_name>` is one of the dataset described in the chosen experiment, and `<path-type` is one of `sp`, `spplus` or `ap` with will use respectively PathNN-$\mathcal{SP}$, PathNN-$\mathcal{SP}^+$ and PathNN-$\mathcal{AP}$.

## Data

---
TUDatasets are available at the following website:

<https://chrsmrrs.github.io/datasets/docs/datasets/>.

Data splits used for the experiments can be found at:

<https://github.com/diningphil/gnn-comparison/tree/master/data_splits>.

If you intend to run experiments on one of the TUDatasets, you can run the script provided in `TUDatasets/datasets/download_data.sh`. This script will download the data from the first URL, the splits from the second URL and move the splits to the subsequent folders.

For all of the datasets used in the `benchmarks` folder, running an experiment with the bash file will directly download and process the data, and the model will be trained directly on the processed data.
Please note that the dataset will be downloaded in the `benchmarks/dataset/` subfolder and will persist after the end of the experiment.

For synthetic datasets, we provide the corresponding text files in this repository and the data will be loaded directly when lauching an experiment.

## Results

---
We provide result logs located in subfolders `<experiment_name>/logs/` for `TUDatasets` and `benchmarks`.

After successfully running an experiment, three subfolders will be created:

- `logs`, containing logs and results of the experiment.
- `results`, containing a serialized python dictionary filled with the experiment results.
- `models`, containing the PyTorch state of the last trained model.

Results can be looked at either on the experiment log file, or by opening the serialized python dictionary directly in Python.

## Cite

If you happen to use or modify this code, please remember to cite our paper:

```
@inproceedings{michel_expressive_2023,
    title = {Path Neural Networks: Expressive and Accurate Graph Neural Networks},
    booktitle = {Proceedings of the 40th {International} {Conference} on {Machine} {Learning} ({ICML})},
    author = {Michel, Gaspard and Nikolentzos, Giannis and Lutzeyer, Johannes and Vazirgiannis, Michalis},
    year = {2023}
}
```
