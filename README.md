# CSGNN
Codes of paper "Category-aware Self-supervised Graph Neural Network for Session-based Recommendation"


## Requirements
- Python 3
- Python 1.6.0

you can create a new environment by conda
```shell
conda install --yes --file requirements.txt
```

## Usage

### datasets
1. directory  `/datasets/id/` is the dataset of CSGNN model 
2. directory `/datasets/pre_training/` is the embeddings of pre-training
3. directories `/datasets/filter*` are the datasets of ablation experiments


  
### Train and evaluate
```shell
python main.py --dataset nowplaying --beta 0.005 --embSize 100 > result.output 
```
