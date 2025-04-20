# MolMVC
MolMVC: a molecular representation learning framework via multi-view contrastive learning for enhancing drug-related tasks

## Pretrained model
You can find our pretrained MolMVC in `/save`. 
`downstream_until.py` can help you generate MolMVC molecular representation and `finetune_MPP.py` can be used as a reference. 

We updated the code on how to use our model to generate molecular representations. You can generate your own molecular representations by `python get_pre_emb.py.`


## Data
Because the file size exceeds the limit, the pretraining dataset can be downloaded from [data](https://drive.google.com/file/d/1fws4GavSfXMlEdh_fV7oTsUSyhSM2lRD/view?usp=sharing). The MPP task benchmark datasets can be downloaded by `python finetune_MPP.py`. The subword of ESPF is in `/ESPF`.


## Environment
You can create a conda environment for MolMVC by `conda env create -f environment.yml`.


## Pretrain and Finetune

You can pretrain MolMVC by `python pretrain_MolMVC.py`. You can finetune the pretrained MolMVC for MPP tasks by `python finetune_MPP.py`.

