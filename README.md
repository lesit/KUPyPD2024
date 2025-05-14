# KUPyPD2024
Code for the paper: **KUPyPD2024: A Knowledge Tracing Benchmark Dataset with Rich Interaction Information for Python Programming Practice.**

## KUPyPD2024 dataset
- The dataset can be downloaded from the link below:
https://huggingface.co/datasets/ini-ku/KUPyPD2024

## Experiments
- Please note that this runs on two GPUs. Modify the shell file if necessary.

### pykt-tookit
#### Attribution

This repository includes code adapted from the following open-source project:
- [pykt](https://github.com/pykt-team/pykt-toolkit), licensed under the MIT License.

#### preprocess
```
python kupypd_preprocess.py --dataset_dir ${download_dir}
```

#### search best hyper-parameters and train
```
sh sh_kupypd_train.sh
```

#### evaluation
```
sh sh_kupypd_eval.sh
```

### Code-DKT & UniXcoder-DKT
#### Attribution

This repository includes code adapted from the following open-source project:
- [Code-DKT](https://github.com/YangAzure/Code-DKT), licensed under the MIT License.
- [UniXcoder](https://github.com/microsoft/CodeBERT), licensed under the MIT License.

#### preprocess
- Please note that default dataset_root_dir is "../../../dataset"
```
sh sh_kupypd_preprocess.sh
```
- If you want to use a different dataset directory, run it as follows:
```
nohup python -u kupypd_preprocess.py --devices 0,1 --dataset_root_dir ${download_dir} > /dev/null 2>&1 &
```

#### search best hyper-parameters
```
sh sh_kupypd_search_hp.sh
```

#### train and evaluation
```
sh sh_kupypd_train_eval.sh
```
