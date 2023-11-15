# NeurIPS 2023 - MedFM: Foundation Model Prompting for Medical Image Classification Challenge 2023

## 🛠️ Installation

Install requirements by

```bash
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
$ pip install mmcls==0.25.0 openmim scipy scikit-learn ftfy regex tqdm
$ mim install mmcv-full==1.6.0
```

### Data preparation

Prepare data following [MMClassification](https://github.com/open-mmlab/mmclassification). The data structure looks like below:

```text
data/
├── MedFMC
│   ├── chest
│   │   ├── images
│   │   ├── chest_X-shot_train_expY.txt
│   │   ├── chest_X-shot_val_expY.txt
│   │   ├── train_20.txt
│   │   ├── val_20.txt
│   │   ├── trainval.txt
│   │   ├── test_WithLabel.txt
│   ├── colon
│   │   ├── images
│   │   ├── colon_X-shot_train_expY.txt
│   │   ├── colon_X-shot_val_expY.txt
│   │   ├── train_20.txt
│   │   ├── val_20.txt
│   │   ├── trainval.txt
│   │   ├── test_WithLabel.txt
│   ├── endo
│   │   ├── images
│   │   ├── endo_X-shot_train_expY.txt
│   │   ├── endo_X-shot_val_expY.txt
│   │   ├── train_20.txt
│   │   ├── val_20.txt
│   │   ├── trainval.txt
│   │   ├── test_WithLabel.txt
```

Noted that the `.txt` files includes data split information for fully supervised learning and few-shot learning tasks.
The public dataset is splited to `trainval.txt` and `test_WithLabel.txt`, and `trainval.txt` is also splited to `train_20.txt` and `val_20.txt` where `20` means the training data makes up 20% of `trainval.txt`.
And the `test_WithoutLabel.txt` of each dataset is validation set.

Corresponding `.txt` files are stored at `./data_backup/` folder, the few-shot learning data split files `{dataset}_{N_shot}-shot_train/val_exp{N_exp}.txt` could also be generated as below:

```shell
python tools/generate_few-shot_file.py
```

Where `N_shot` is 1,5 and 10, respectively, the shot is of patient(i.e., 1-shot means images of certain one patient are all counted as one), not number of images.

The `images` in each dataset folder contains its images, which could be achieved from original dataset.

### Training and evaluation using OpenMMLab codebases.

In this repository we provided many config files for fully supervised task (only uses 20% of original traning set, please check out the `.txt` files which split dataset)
and few-shot learning task.

The config files of fully supervised transfer learning task are stored at `./configs/densenet`, `./configs/efficientnet`, `./configs/vit-base` and
`./configs/swin_transformer` folders, respectively. The config files of few-shot learning task are stored at `./configs/ablation_exp` and `./configs/vit-b16_vpt` folders.


You can generate all prediction results of `endo_N-shot_submission.csv`, `colon_N-shot_submission.csv` and `chest_N-shot_submission.csv` and zip them into `result.zip` file. Then upload it to Grand Challenge website.

```
result/
├── endo_1-shot_submission.csv
├── endo_5-shot_submission.csv
├── endo_10-shot_submission.csv
├── colon_1-shot_submission.csv
├── colon_5-shot_submission.csv
├── colon_10-shot_submission.csv
├── chest_1-shot_submission.csv
├── chest_5-shot_submission.csv
├── chest_10-shot_submission.csv
```
