# C3AE_Age_Estimation


## Introduction
[C3AE: Exploring the Limits of Compact Model for Age Estimation](https://arxiv.org/abs/1904.05059)



This repo is organized as follows:

```
C3AE_Age_Estimation/
    |->examples
    |->models
    |->prepare_data
    |->data
    |   |->img_list
    |   |->dataset
    |->tf_records
    |->ckpt
    |->tools
```

## Requirements
1. tensorflow-gpu==1.12.0 (I only test on tensorflow 1.12.0)
2. python==3.4.3
3. numpy
4. easydict
5. opencv==3.4.1
6. Python packages might missing. pls fix it according to the error message.

## Installation, Prepare data, Training
### Installation
1. Clone the C3AE_Age_Estimation repository, and we'll call the directory that you cloned C3AE_Age_Estimation as `${C3AE_Age_Estimation_ROOT}`.

```
git clone https://github.com/vicwer/C3AE_Age_Estimation.git
```

2. Create data, tf_records and ckpt directory. 

```
cd ${C3AE_Age_Estimation_ROOT};
mkdir ckpt
mkdir tf_records
mkdir data
cd data
mkdir img_list
mkdir train_list
mkdir dataset
```

### Prepare data
data should be organized as follows:

```
data/
    |->img_list/img_list.txt
    |->train_list/train.txt
    |->dataset/*.png
```
1. Download dataset: IMDB-WIKI, Morph II, FG-NET

2. Generate img_list.txt formatted as "img_path age"

3. Generate train.txt formatted as "img_path age_label age_Yn_vector"

4. Generate tf_records:

```
cd prepare_data
python3 gen_tf_records_fast_to_uint8.py
```

### Training

I provide common used config.py in ${C3AE_Age_Estimation_ROOT}, which can set hyperparameters.

e.g.
```
cd ${C3AE_Age_Estimation_ROOT}
vim config.py
cfg.train.num_gpus = {your gpu nums}
etc.

cd ${C3AE_Age_Estimation_ROOT}/examples/
python3 multi_gpus_train.py
```

## TODO:

```
test.py
tools.py
pre-train model
...
```

## GOOD LUCK...
