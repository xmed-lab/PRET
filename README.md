# PRET: Achieving Pan-Cancer Recognition via a Few Samples Without Training


## Introduction

**PRET** (**P**an-cancer **R**ecognition without **E**xample **T**raining) is an innovative approach for multi-cancer diagnostics and tasks that eliminates task-specific model training. Utilizing a few labeled examples, PRET empowers pathological foundation models with the capability to directly recognize pan-cancer in the manner of in-context learning (ICL) to learn from inference stage. PRET fully accounts for the unique characteristics of whole slide images, where massive patch tiles preserve rich local information, thereby facilitating exceptional recognition capabilities. By offering a flexible and cost-effective solution for pan-cancer recognition, PRET paves the way for accessible and equitable AI-based pathology systems, particularly benefiting minority populations and underserved regions.

* This the data flow of our method to assist in understanding the code.

![](https://github.com/xmed-lab/PRET/blob/main/preview.png)

* This repo is originally relased at ([HF Code](https://huggingface.co/yili7eli/PRET)).

* Besides the code, all in-house datasets and labels are available too ([HF DATA](https://huggingface.co/datasets/yili7eli/PRET/tree/main)).

* **The paper will be released upon acceptance.**

## Requesting Access
As required by our data providers, you must agree to the outlined terms of use and provide the required information for approval. We will approve your request as soon as possible so that you can download the code model and data. Other reasons for your request access being denied include other mistakes in the form submitted, for example: full name includes abbreviations, affiliation is not spelled out, the described research use is not sufficient, or email domain address not recognized. We close the approval process upon paper acceptance.


## Download Resources
* After request, login with your Access Token for next steps.
```
pip install --upgrade huggingface_hub
huggingface-cli login
```

* Download this repo for the code:
```
git clone git@github.com:xmed-lab/PRET.git
```

* Download our datasets and labels in PRET/data
```
cd PRET
huggingface-cli download yili7eli/PRET --local-dir data --repo-type dataset
```

* The default foundation model is provided in this repo. If other foundation models are needed, please download their extra model, code, and dependencies following their websites. Their feature extraction pipelines are included in the core/feature_extractor.py.

* Download public datasets from their official websites and save slides in folders like "data/NSCLC/images".


## Install
* Install python packages via pip:
```
cd PRET
pip install -r requirements.txt
```

* Install libvips via apt (ubuntu) for WSI slicing:
```
sudo apt install libvips-dev
```


## Dataset Process
Slice WSIs to patches; process annotations; generate data information (given in data_info/).
* For In-house dataset and TCGA datasets:
```
#bash scripts/prepare_dataset.sh [DATASET_NAME]; for example:
bash scripts/prepare_dataset.sh ESCC
```
* For CAMELYON datasets:
```
#bash scripts/prepare_dataset_camelyon.sh [DATASET_NAME]; for example:
bash scripts/prepare_dataset_camelyon.sh CAMELYON16
```


## Batch Evaluation
Run scripts/batch_run.py for convenient batch evaluation after dataset processing.

* The batch evaluation script run all involved tasks, prompt types, repeat experiments for a dataset.
* The script run multiple experiments at once, the number of repeat experiments is n_tasks x n_prompts x n_repeats.

```
#python scripts/batch_run.py [DATASET_NAME] [MODEL_NAME_OR_WEIGHTS] [PARALLEL_TASK_NUM]
python scripts/batch_run.py ESCC model.pth 4
```

## Single Evaluation
Run scripts/run.py for a single benchmark.

* Run a single benchmark by assigned task, prompt.
* The number of repeat experiment is n_repeats only.
* The modes include default, baselines, eval (wo. hyperparameter search).
```
#python scripts/run.py [GPU_ID] [DATASET_NAME] [TASK] [MODE] [PROMPT_TYPE] [MODEL_NAME_OR_WEIGHTS]
python scripts/run.py 0 ESCC screening default slideLabel model.pth
```


## Code Explanation
Code flow: a. scripts/batch_run.py -> b. core/feature_extractor.py -> c. scripts/run.py -> d. core/main.py -> e. core/modules.py


* a. scripts/batch_run.py: It first extracts feature by invoking the core/feature_extractor.py as:
```python
# extract feature by invoking the core/feature_extractor.py (line 73)
if not os.path.exists(FEAT_DIR):
    command = 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, visible_devices)) + ' python -u -m torch.distributed.launch --nproc_per_node=' + str(len(visible_devices)) + ' --master_port=2333 core/feature_extractor.py ' + ' --data_path ' + PATCH_DIR + ' --pretrained_weights ' + model + ' --dump_features ' + FEAT_DIR + ' --num_workers 8 --batch_size_per_gpu 32 --arch ' + arch
os.system(command)
```

* a. scripts/batch_run.py: Then, it gets commands from core/feature_extractor.py for each task, prompt, and mode as:
```python
# invoke the scripts/run.py for the command for each benchmarks (line 99)
commands.append(get_run_command(dataset, task, mode, prompt_type, model))
# excute commands using multiprocessing.Pool with GPU management (line 116)
pool.apply_async(execute_bash_command, args=(command, out_text), kwds={}, callback=None)
```

* b. core/feature_extractor.py: It first loads the model, supporting diverse foundation models:
```python
# We support many foundation models. Here we just show two examples for reference.
# We havd provided the weights of default model, others need to download their code, dependencies and weights following their official websites.
# load the default model from local weights
if "vit_small" in args.arch:
    sys.path.append("network")
    import vision_transformer as vits
    model = vits.__dict__[args.arch](patch_size=8, num_classes=0)
    utils.load_pretrained_weights(model, args.pretrained_weights)
    print(f"Model {args.arch} 8x8 built.")
# load from hugging face, you are required to clone the code in the home dir with its dependencies
elif args.arch == 'conch':
    sys.path.append(os.path.expanduser('~/CONCH'))
    from conch.open_clip_custom import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.pretrained_weights)
...
```

* b. core/feature_extractor.py: Then, it extracts features for patch tiles according to the inference codes of FMs:
```python
# two mode, patch feature or zero-shot features
if arch == 'conch':
    feats = model.encode_image(imgs, proj_contrast=False, normalize=False)
    #feats = model.encode_image(imgs, proj_contrast=True, normalize=False) # for zero-shot
# use its own function
elif arch == 'plip':
    feats = model.get_image_features(imgs)
# virchow needs feature concat
elif arch == 'virchow':
    feats = model(imgs)
    class_token = feats[:, 0]
    patch_tokens = feats[:, 1:]
    feats = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
# used for zero-shot only
elif arch == 'musk':
    feats = model(image=imgs.half(), with_head=True, out_norm=False)[0]
# other models, directly apply the forward function
else:
    feats = model(imgs)
```

* c. core/run.py: It contains basic information for benchmarks including folders, hyper-parameters, data split, with detailed comments as:
```python
# ===================== set data split ======================
# data split principle
# 1) TCGA 20% test similating 5-fold, 100 val or 50 val if not sufficient
# 2) New datasets is smaller than TCGA, thus 1/3 val 2/3 test for slides out of example
# 3) Cross race NSCLC, White follow TCGA split, others follow new datasets
```

* c. core/run.py: Then, it genetates the running command for core/main.py with args:
```
# ===================== generate the command ======================
# prepare args for the core/main.py
command = 'python -u core/main.py --mode ' + mode + ' --topk ' + str(TOPK) + ' --temperature ' + str(TEMP) + ' --related_thresh ' + str(THRESH) + ' --example_num ' + str(SHOT_NUM) + ' --raw_feature_path ' + FEAT_DIR + ' --wsi_path '  + WSI_DIR + ' --dump_features ' + COLLECTED_FEAT_DIR + ' --dataset_info data_info/' + dataset + '.json --seed ' + str(SEED) + ' --top_instance ' + str(TOP_INS) + ' --test_num ' + str(TEST_NUM) + ' --val_num ' + str(VAL_NUM) + ' --val_ratio ' + str(VAL_RATIO) + ' --prompt_type ' + prompt_type + ' --prompt_path ' + PROMPT_DIR + ' --ignore ' + str(IGNORE) + ' --multiple_num 1 2 4 8 --file_min_size ' + str(FILE_MIN_SIZE) + ' --c ' + str(CLS_NUM) + ' --ignore_query ' + str(IGNORE_QUERY) + ' --dump_records ' + RECORDS + '.npy'
if task == 'segmentation':
    command += ' --seg '
# save the records in both .txt and .npy formats
command += ' &> ' + RECORDS + '.txt'
```
* d. core/main.py: It is responsible for parsing parameters, collecting features, selecting models, and then evaluating PRET and baselines. The code function is "evaluation", where we have provide many comments to explain the code parts, see comments below:
```python
# ====================== repeat experimets n=args.runs ====================== (line 158)
# ====================== data split ====================== (line 163)
# ====================== run for each class ====================== (line 252)
# ====================== process example and prompts ====================== (line 257)
# ====================== apply in-context tagger ====================== (line 316)
# ====================== predict for test slides (queries)====================== (line 357)
# ====================== discriminative instance miner for subtyping ====================== (line 371)
# ====================== inference, including classifier, aggregator, post processer ====================== (line 379)
# ====================== process validation set and assign label ====================== (line 411)
# ====================== count and record results ====================== (line 468)
```

* e. core/modules: The core modules are implemented in this file, including the in-context tagger, classifier, miner, aggregator, and post processor. We allocat these modules with comments as follows:
```python
# ====================== instance miner for subtyping ====================== (line 226)
# ====================== basic tagger (algorithm 1) ====================== (line 251)
# ====================== in-context tagger (algorithm 2) ====================== (line 273)
# ====================== in-context tagger for subtyping (algorithm 2) ====================== (line 327)
# ====================== in-context classifier ====================== (line 399)
# ====================== attention aggregator ====================== (line 418)
# ====================== patch reshape to wsi for heatmap / seg ====================== (line 437)
# ====================== segmentation post processor ====================== (line 459)
```


## Record Examples
We have saved some record examples in the records folder to prove the reproducibility.
* The results of ESCC screening with slide label are recorded as follows:
```
eval 1-shot:
class:1 val auc:0.9919, test auc:0.9577, val acc: 0.9375, test f1: 0.8861, test acc: 0.8636
class:1 val auc:0.9762, test auc:0.9639, val acc: 0.9062, test f1: 0.8861, test acc: 0.8636
class:1 val auc:0.9028, test auc:0.9481, val acc: 0.8438, test f1: 0.9268, test acc: 0.9091
class:1 val auc:0.8988, test auc:0.9279, val acc: 0.8438, test f1: 0.8649, test acc: 0.8485
class:1 val auc:0.8785, test auc:0.899, val acc: 0.8125, test f1: 0.8605, test acc: 0.8182
auc mean: 0.9393, auc std: 0.0235, f1 mean: 0.8849, f1 std: 0.0235, acc mean: 0.8606, acc std: 0.0294
eval 2-shot:
class:1 val auc:0.9603, test auc:0.9563, val acc: 0.9375, test f1: 0.8916, test acc: 0.8594
class:1 val auc:0.9484, test auc:0.9823, val acc: 0.875, test f1: 0.9067, test acc: 0.8906
class:1 val auc:0.8696, test auc:0.933, val acc: 0.8438, test f1: 0.8101, test acc: 0.7656
class:1 val auc:0.9818, test auc:0.9375, val acc: 0.9375, test f1: 0.8986, test acc: 0.8906
class:1 val auc:0.9883, test auc:0.9643, val acc: 0.9688, test f1: 0.8889, test acc: 0.8594
auc mean: 0.9547, auc std: 0.018, f1 mean: 0.8792, f1 std: 0.0351, acc mean: 0.8531, acc std: 0.0459
eval 4-shot:
class:1 val auc:0.942, test auc:0.9393, val acc: 0.9333, test f1: 0.9048, test acc: 0.871
class:1 val auc:0.9689, test auc:0.9697, val acc: 0.9333, test f1: 0.8421, test acc: 0.8065
class:1 val auc:0.9911, test auc:0.969, val acc: 0.9333, test f1: 0.9136, test acc: 0.8871
class:1 val auc:0.97, test auc:0.9081, val acc: 0.9333, test f1: 0.8571, test acc: 0.8226
class:1 val auc:0.9861, test auc:0.9912, val acc: 0.9667, test f1: 0.925, test acc: 0.9032
auc mean: 0.9555, auc std: 0.0289, f1 mean: 0.8885, f1 std: 0.0327, acc mean: 0.8581, acc std: 0.0373
eval 8-shot:
class:1 val auc:0.9893, test auc:0.9388, val acc: 0.9643, test f1: 0.8788, test acc: 0.8571
class:1 val auc:1.0, test auc:0.9864, val acc: 0.9643, test f1: 0.9855, test acc: 0.9821
class:1 val auc:0.9875, test auc:0.9857, val acc: 0.9286, test f1: 0.9143, test acc: 0.8929
class:1 val auc:0.9883, test auc:0.9829, val acc: 0.9286, test f1: 0.9375, test acc: 0.9286
class:1 val auc:0.9883, test auc:0.9802, val acc: 0.9286, test f1: 0.8462, test acc: 0.7857
auc mean: 0.9748, auc std: 0.0181, f1 mean: 0.9124, f1 std: 0.048, acc mean: 0.8893, acc std: 0.0662
```
* More detailed records are saved in the .npy files, including the data split, logits, predictions, and all results.


## Other Notes

### Dataset Files
* The class.json contains the class names (e.g. {"Adenocarcinoma, NOS": 1, "Squamous cell carcinoma, NOS": 2}).
* The label.txt contains file names with slide labels (e.g. xxx.svs,1).
* The images and anno folders storage WSIs and annotations (via software ImageScope), respectively.

### Script Explaination:
* The scripts/batch_run.py firstly invokes core/feature_extractor.py for feature extraction. Then it generate running commands for target datasets, methods, prompts, and tasks.
* The scripts/run.py generates command to invoke the core/main.py to run a specific task with repeat experiments (e.g., ESCC-screening-imgLabel).
* The core/feature_extractor.py extract patch features, we have implemented some other foundation models. Some models require to save their code in the home folder, and some models use huggingface to download.
* The core/modules.py contains the implementation of core modules, including tagger, miner, classifier, aggregator, pos pocessor.
* The core/main.py is the main code, involving our method and baselines to evaluate multiple shot and repeats.

### Extensive Datasets
* The CAMELYON16-C simulates scan corruptions using the same slides of CAMELYON16, uncommand "RandomDistortions" in the core/feature_extractor.py to activate it. Besides, copy the dataset and data_info file from CAEMLYON16 to CAEMLYON16C.
* We provide the label.cvs for CAMELYON17 about the data source and LNM size. External experiments in Fig.5 use micro and macro LNM as CAMELYON16.
* Experiments about external prompts needs to combine multiple datasets. You can create a new dataset (e.g., data/PTC_QP2GD) and merge their files. Besides, set the "fixed_test_set" (e.g., in data_info/PTC_QP2GD.json) to true for the external hospital.

### Reproducibility
* The data_info provides a fixed data list. We change get a fixed data split with seed 1024 to ensure the same examples, val slides and test slides.
* The different package versions may slightly change the results (within an acceptable range), including scikit-learn, CUDA, torch, torchvision, cv2, pillow, etc., that related to evaluation and data loading.

### Time Cost
* Package installation usually takes less than an hour depending on network speed.
* Dataset downloading takes hours or a few days depending on network speed (over 800GB for our in-house datasets).
* The process involves slide slicing, feature extraction, hyperparameter search, multiple shot settings, multiple prompts, and repeated experiments, which can take a few or dozens of hours depending on the data scale, CPU, GPU, and IO speed;

## Citation

The paper is coming soon (accepted in principle currently).

## License
```
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```
