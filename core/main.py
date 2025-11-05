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

import os
import sys
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import numpy as np
import cv2
import openslide
from sklearn.metrics import precision_recall_curve

from modules import inference, load_weak_prompts, execute_tagger, \
        execute_subtyping_tagger, execute_miner


# ====================== collect features and information ======================

def feature_processor(args):
    print('start feature processing ...')
    dataset_info = json.load(open(args.dataset_info))
    os.makedirs(args.dump_features, exist_ok=True)

    for k, v in dataset_info.items():
        if os.path.exists(os.path.join(args.dump_features, k + '.npy')):
            continue

        feats, names, patch_label, wsi_label = [], [], [], -1
        
        wsi_label = v['wsi_label']

        # patch label as segmentation gt, if any
        if 'patch_labels' in v:
            mask = cv2.imread(v['patch_labels'])[:, :, 0]

        in_dir = os.path.join(args.raw_feature_path, k + '_files')
        in_dir = in_dir if in_dir[-1] != '/' else in_dir[:-1]
        patch_path = in_dir.replace(in_dir.split('/')[-2], 'images')
        ori_dir = sorted([int(_) for _ in os.listdir(patch_path)])[-1]
        patch_path = os.path.join(patch_path, str(ori_dir))

        for f in os.listdir(os.path.join(in_dir, 'x20')):
            name = os.path.join(patch_path, f.replace('.npy', '.jpeg'))
            if os.path.getsize(name) < args.file_min_size:
                continue
            
            # load feature, L2 norm
            feat = np.load(os.path.join(in_dir, 'x20', f))
            feat = feat / np.linalg.norm(feat, ord=2, axis=0)

            # get position for vis and seg
            x, y = f.split('.')[0].split('_')
            x, y = int(x), int(y)
            if 'patch_labels' in v:
                patch_label.append(mask[y, x])

            names.append(name)
            feats.append(feat)
        
        if len(names) == 0:
            continue

        # save patch features, name, patch_labels and wsi_labels for eval
        info = {'features': np.stack(feats, 0), 'patch_names': names, \
            'patch_labels': np.array(patch_label), 'wsi_label': wsi_label}
        np.save(os.path.join(args.dump_features, k + '.npy'), info)
    
    print('finish feature processing and saving!')


# ====================== some util functions ======================

def macro_value(l, n):
    out = []
    for i in range(len(l) // n):
        v = sum(l[i * n: i * n + n]) / n
        out.append(v)
    return out


def get_example_names_at_same_num(all_names, dataset_info, example_num, check_num=False):
    record = {}
    for n in all_names:
        lb = dataset_info[n]['wsi_label']
        if lb not in record:
            record[lb] = []
        record[lb].append(n)

    names = []
    for k, v in record.items():
        if check_num == True and len(v) < (example_num):
            print('exist! insufficient samples. ' + str(k))
            sys.exit(0)
        names.extend(v[:example_num])

    return names


def check_different_patient(example_names, query_candidates, mode='TCGA'):
    out = []
    if mode == 'TCGA':
        for q in query_candidates:
            inside = False
            for g in example_names:
                if mode == 'TCGA':
                    if q[:12] == g[:12]:
                        inside = True
            if not inside:
                out.append(q)

    return out


# post processing via gussain blur
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(GaussianBlur, self).__init__()
        kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2)/(2*sigma**2)),
        (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        kernel = np.reshape(kernel, (1, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(torch.from_numpy(kernel).float(), requires_grad=False).cuda()

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.weight.shape[-1]//2)


# ====================== evaluation for multiple tasks, prompts, and shots ======================

def evaluate(args, val_only=False):
    auc_list, f1_list, acc_list, example_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    info_str = open(args.dataset_info).read()
    dataset_info = json.load(open(args.dataset_info))
    all_names = dataset_info.keys()

    records = {}
    txt_rec = []
    # ====================== repeat experimets n=args.runs ======================

    for i in range(args.runs):
        records['repeat_' + str(i)] = {}
        
        # ====================== data split ======================

        labeled_names, neg_names, test_names, rest_names = [], [], [], []

        for n in all_names:
            # splitdata, if there is fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)

            else:
                # pick pos from labeled wsi
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']
                    
                    # prompt samplinging (camelyon only)
                    if args.c == 1 and 'CAMELYON' in args.wsi_path:
                        if pn >= 1000 and pn < 3000:
                            labeled_names.append(n)
                    
                    else:
                        labeled_names.append(n)
                
                if args.prompt_type == 'slideLabel':
                    # add neg and pos for subtyping (no labeled wsis)
                    if args.c > 1 and 'pos_patch_num' not in info_str:
                        labeled_names.append(n)
                
                    # add some neg for slideLabel binary cls
                    if args.c == 1 and dataset_info[n]['wsi_label'] == 0:
                        labeled_names.append(n)
                
                # record neg names to exclude from seg val /test
                if dataset_info[n]['wsi_label'] == 0:
                    neg_names.append(n)

        # shuffle example till each run is different
        while True:
            random.shuffle(labeled_names)

            # randomly select "args.example_num" examples for each class
            # note: for binary tasks 'slideLabel' use N // 2 pos and N // 2 neg
            if args.c > 1 or args.prompt_type == 'slideLabel':
                example_i = get_example_names_at_same_num(labeled_names, dataset_info, args.example_num, args.c > 1)

            # randomly select "args.example_num" positive examples for binary tasks
            else:
                example_i = labeled_names[:args.example_num]

            # avoid repeat example
            example_i.sort()
            if example_i not in example_list:
                example_list.append(example_i)
                example_names = example_i
                break

        # split val set out of example and test set
        for n in all_names:
            if n not in example_names and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)

        if args.seg:
            rest_names = []
            for ln in labeled_names:
                if ln not in example_names and ln not in neg_names:
                    rest_names.append(ln)

        # avoid same patients in different split, in-house data is cleaned
        if 'TCGA' in args.wsi_path:
            rest_names = check_different_patient(example_names, rest_names, 'TCGA')

        random.shuffle(rest_names)
        val_num = args.val_num if args.val_ratio < 0 else int(len(rest_names) * args.val_ratio)
        val_names = rest_names[:val_num]

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            if args.val_ratio < 0:
                test_names = rest_names[-args.test_num:]
            else:
                test_names = rest_names[val_num:]
            if len(val_names) + len(test_names) > len(rest_names):
                print('wrong split size !!!')
        else: # take partial test slides for tcga cross races
            random.shuffle(test_names)
            if args.test_num > 0:
                test_names = test_names[:args.test_num]
        
        records['repeat_' + str(i)]['split'] = {'example_names': example_names, 'val_names': val_names, 'test_names': test_names}

        # ====================== run for each class ======================

        # for subtyping, use different example for each cls and apply marco metics, other tasks have one class
        for cls in range(1, args.c + 1):

            # ====================== process example and prompts ======================

            # load example
            example_feats, example_patch_names, example_labels = [], [], []
            for n in example_names:
                example_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                example_patch_names = example_patch_names + example_n['patch_names']
                example_feats.append(example_n['features'])

                # empty patch label for image label or sparse label where there is no offline gt
                if args.prompt_type == 'mask':
                    pl = example_n['patch_labels']

                    # binary use 0 normal, 1 tumor, while subtyping use 0 other cls, 1 this cls, 255 normal
                    if args.c > 1:
                        pl[pl == 0] = 255
                        if example_n['wsi_label'] != cls:
                            pl[pl == 1] = 0
                        else:
                            pl[pl == 1] = 1
                    
                else:
                    pl = np.zeros(example_n['features'].shape[0]) - 1
                
                # load weak prompts
                # slideLabel + subtyping is uniqe in pseudo label generation
                if args.prompt_type == "slideLabel" and args.c > 1:
                    if example_n['wsi_label'] != cls:
                        pl[:] = 0
                    else:
                        pl[:] = 1

                # for box, RoughMask and binary + slideLabel, -1 is uncertain pos, 0 is normal
                elif args.prompt_type != 'mask' :
                    pl = load_weak_prompts(n, example_n['wsi_label'], args.wsi_path, pl, \
                        example_n['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale)
                    
                    #  record wsi label for each patch for later label convert
                    if args.c > 1:
                        pl[pl == 0] = 255
                        pl[pl == -1] = 1 if example_n['wsi_label'] == cls else 0

                example_labels.append(pl)
            
            example_feats = torch.tensor(np.concatenate(example_feats, 0)).cuda()
            example_labels = torch.tensor(np.concatenate(example_labels, 0)).cuda().long()

            if args.dump_pseudo != '':
                vis_info = {'wsi_dir': args.wsi_path, 'vis_dir': os.path.join(args.dump_pseudo, 'vis') + str(args.example_num) + '/' + str(i) + '/' + str(cls), \
                        'mask_dir': os.path.join(args.dump_pseudo, 'pseudo') + str(args.example_num) + '/' + str(i) + '/' + str(cls)}

                split_dir = os.path.join(args.dump_pseudo, 'split') + str(args.example_num)
                split = {'example_names': example_names, 'test_names': test_names, 'val_names': val_names}
                os.makedirs(split_dir, exist_ok=True)
                open(os.path.join(split_dir, str(i) + '.json'), 'w').write(json.dumps(split, indent=4))

            else:
                vis_info = None

            # ====================== apply in-context tagger ======================

            # assign in-context tags for weak prompts (binary tasks: 1 pos, 0 neg, -1 unknown)
            if args.prompt_type != 'mask' and args.c == 1:
                example_labels = execute_tagger(example_feats, example_labels, example_patch_names, example_names, \
                    vis_info=vis_info, uncertain=args.ignore, topk=args.topk)

            # assign in-context tags for subtyping from slideLabel (255 normal, 254 uncertain, 1 this class, 0 other classes)
            if args.prompt_type == 'slideLabel' and args.c > 1:
                example_labels = execute_subtyping_tagger(example_feats, example_labels, example_patch_names, \
                    example_names, vis_info=vis_info, uncertain=args.ignore, topk=args.topk)
            
            # subtyping + box / roughMask. Need to process "execute_tagger" twice. 
            # Once for shared bg and this class, another for shared bg and other classes
            if args.prompt_type != 'slideLabel' and args.c > 1:
                
                if args.prompt_type != 'mask':
                    example_labels_this = example_labels.clone()
                    example_labels_this[example_labels_this == 0] = 254 # ignore other fg
                    example_labels_this[example_labels_this == 255] = 0 # subtyping bg label to binary neg label
                    example_labels_this[example_labels_this == 1] = -1  # this class to undertain to relabel
                    example_labels_this = execute_tagger(example_feats, example_labels_this, example_patch_names, example_names, \
                        vis_info=vis_info, uncertain=args.ignore, topk=args.topk)

                    vis_info = None
                    example_labels_others = example_labels.clone()
                    example_labels_others[example_labels_others == 1] = 254 # ignore this fg
                    example_labels_others[example_labels_others == 0] = -1  # other class to undertain to relabel
                    example_labels_others[example_labels_others == 255] = 0 # subtyping bg label to binary neg label
                    example_labels_others = execute_tagger(example_feats, example_labels_others, example_patch_names, example_names, \
                        vis_info=vis_info, uncertain=args.ignore, topk=args.topk)

                    example_labels[:] = 255 # default bg
                    example_labels[example_labels_this == 1] = 1     # this class
                    example_labels[example_labels_others == 1] = 0   # other class
                    example_labels[example_labels_this == -1] = 254  # ignore in the last
                    example_labels[example_labels_others == -1] = 254# ignore in the last
                    if (example_labels == 255).sum() == 0:
                        example_labels[example_labels_others == 0] = 255
                        example_labels[example_labels_this == 0] = 255

            # ====================== predict for test slides (queries)======================

            # predict for test slides, name a test slide as query to avoid confusion with test set
            val_preds, test_preds, val_labels, test_labels = [], [], [], []
            wsi_suffix = os.listdir(args.wsi_path)[0].split('.')[-1]
            all_query_names = val_names if val_only else val_names + test_names
            for n in all_query_names:
                query_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                query_feats = torch.tensor(query_n['features']).cuda()
                query_patch_names = query_n['patch_names']
                label = query_n['wsi_label']
                if args.c > 1:
                    label = int(label == cls)
                
                # ====================== discriminative instance miner for subtyping ======================

                # use fg patches for subtyping
                #if args.c > 1 and not args.seg and args.vis_path == '': # vis wo fg
                if args.c > 1 and not args.seg:
                    query_feats, query_patch_names = execute_miner(example_feats[example_labels == 255], \
                        query_feats, query_patch_names, uncertain=args.ignore_query)

                # ====================== inference, including classifier, aggregator, post processer ======================

                wsi_path = os.path.join(args.wsi_path, n + '.' + wsi_suffix)
                if os.path.exists(wsi_path):
                    wsi = openslide.OpenSlide(wsi_path)
                    size = (wsi.level_dimensions[0][1] // args.patch_scale, wsi.level_dimensions[0][0] // args.patch_scale)
                else:
                    size = None
                vis_info = None
                
                sm = GaussianBlur(7, 3) if args.seg else None #  seg pred
                wsi_pred, patch_pred, patch_pred_list = inference(args, example_feats, example_labels, example_patch_names, \
                    query_feats, query_patch_names, size, args.top_instance, vis_info, smooth=sm)

                if patch_pred != None and args.vis_path != '' and n in test_names:
                    os.makedirs(args.vis_path, exist_ok=True)
                    np.save(os.path.join(args.vis_path, n + '_' + str(cls) + '.npy'), patch_pred.cpu().numpy())
                
                if args.seg:
                    pred = torch.tensor(patch_pred_list)
                    label = torch.tensor(query_n['patch_labels'])
                else:
                    pred = torch.tensor([wsi_pred])
                    label = torch.tensor([label])

                if n in val_names:
                    val_preds.append(pred)
                    val_labels.append(label)
                else:
                    test_preds.append(pred)
                    test_labels.append(label)

            # ====================== process validation set and assign label ======================

            # Evaluate on the val set to make sure qualified results for application
            # Val set also guidances to select prediction threshod, f1 for seg. acc for others
            val_preds = torch.cat(val_preds).cpu()
            val_labels = torch.cat(val_labels)
            val_auc = aucroc(val_preds, val_labels).item()
            if not val_only:
                test_preds = torch.cat(test_preds).cpu()
                test_labels = torch.cat(test_labels)
            
            precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
            accs = np.array([((val_preds > _).float() == val_labels).sum() / val_labels.shape[0] for _ in thresholds])
            if args.seg:
                f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_acc_score = accs[best_f1_score_index]
                thresh = thresholds[best_f1_score_index]
            else:
                best_acc_score = np.max(accs[np.isfinite(accs)])
                best_acc_score_index = np.argmax(accs[np.isfinite(accs)])
                thresh = thresholds[best_acc_score_index]

            if val_only:
                preds = val_preds
                thresh_preds = (val_preds > thresh).float()
                labels = val_labels
            else:
                preds = test_preds
                thresh_preds = (test_preds > thresh).float()
                labels = test_labels

            acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
            rec = ((thresh_preds * labels).sum() / labels.sum()).cpu().item()
            pre = ((thresh_preds * labels).sum() / thresh_preds.sum()).cpu().item()
            auc = aucroc(preds, labels).item()
            if (rec + pre) != 0:
                f1 = rec * pre * 2 / (rec + pre)
            else:
                f1 = 0
            auc_list.append(auc)
            f1_list.append(f1)
            acc_list.append(acc)
            if not val_only:
                s = 'class:' + str(cls) + ' val auc:' + str(round(val_auc, 4)) + ', test auc:' + str(round(auc, 4)) + \
                    ', val acc: ' + str(round(best_acc_score, 4)) + ', test f1: ' + str(round(f1, 4)) + \
                    ', test acc: ' + str(round(acc, 4))
                print(s)
                txt_rec.append(s)
                records['repeat_' + str(i)]['results_cls' + str(cls)] = {'val_auc': round(val_auc, 4), 'test_auc': round(auc, 4), \
                        'val_acc': round(best_acc_score, 4), 'test_f1': round(f1, 4), 'test_acc': round(acc, 4)}
                records['repeat_' + str(i)]['pred_cls' + str(cls)] = {'labels': labels.cpu().tolist(), \
                        'logits': preds.cpu().tolist(), 'preds': thresh_preds.cpu().tolist()}

        del example_feats, query_feats
        torch.cuda.empty_cache()

    # ====================== count and record results ======================

    auc_mean = np.array(auc_list).mean()
    macro_auc = macro_value(auc_list, args.c)
    auc_std = np.array(macro_auc).std()
    f1_mean = np.array(f1_list).mean()
    macro_f1 = macro_value(f1_list, args.c)
    f1_std = np.array(macro_f1).std()
    acc_mean = np.array(acc_list).mean()
    macro_acc = macro_value(acc_list, args.c)
    acc_std = np.array(macro_acc).std()
    s = 'auc mean: ' + str(round(auc_mean, 4)) + ', auc std: ' + str(round(auc_std, 4)) + \
        ', f1 mean: ' + str(round(f1_mean, 4)) + ', f1 std: ' + str(round(f1_std, 4)) + \
        ', acc mean: ' + str(round(acc_mean, 4)) + ', acc std: ' + str(round(acc_std, 4))
    print(s)
    txt_rec.append(s)

    records['mean'] = {'auc_mean': round(auc_mean, 4), 'auc_std': round(auc_std, 4), 'auc_values': macro_auc, \
            'f1_mean': round(f1_mean, 4), 'f1_std': round(f1_std, 4), 'f1_values': macro_f1, \
            'acc_mean': round(acc_mean, 4), 'acc_std': round(acc_std, 4), 'acc_values': macro_acc}
    records['text_records'] = txt_rec
     
    return round(auc_mean, 4), records


# ====================== evaluation for baseline methods ======================

def evaluate_baseline(args, mode):
    auc_list, f1_list, acc_list, example_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    info_str = open(args.dataset_info).read()
    dataset_info = json.load(open(args.dataset_info))
    all_names = dataset_info.keys()

    # skip invalid wsis
    temp = []
    for _ in all_names:
        if os.path.exists(os.path.join(args.dump_features, _ + '.npy')):
            temp.append(_)
    all_names = temp

    # ====================== run for each class ======================

    records = {}
    txt_rec = []
    for i in range(args.runs):
        records['repeat_' + str(i)] = {}

        # ====================== data split ======================

        # data split
        labeled_names, neg_names, test_names, rest_names = [], [], [], []

        for n in all_names:
            # splitdata, if there is fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)

            else:
                # pick pos from labeled wsi
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']

                    # prompt samplinging (camelyon only)
                    if args.c == 1 and 'CAMELYON' in args.wsi_path:
                        if pn >= 1000 and pn < 3000:
                            labeled_names.append(n)

                    else:
                        labeled_names.append(n)

                if args.prompt_type == 'slideLabel':
                    # add neg and pos for subtyping (no labeled wsis)
                    if args.c > 1 and 'pos_patch_num' not in info_str:
                        labeled_names.append(n)

                    # add some neg for slideLabel binary tasks
                    if args.c == 1 and dataset_info[n]['wsi_label'] == 0:
                        labeled_names.append(n)

                # record neg names to exclude from seg val /test
                if dataset_info[n]['wsi_label'] == 0:
                    neg_names.append(n)

        # shuffle example till each run is different
        while True:
            random.shuffle(labeled_names)

            # randomly select "args.example_num" examples for each class
            # note: for binary tasks 'slideLabel' use N // 2 pos and N // 2 neg
            if args.c > 1 or args.prompt_type == 'slideLabel':
                example_i = get_example_names_at_same_num(labeled_names, dataset_info, args.example_num, args.c > 1)

            # randomly select "args.example_num" positive examples for binary tasks
            else:
                example_i = labeled_names[:args.example_num]

            # avoid repeat example
            example_i.sort()
            if example_i not in example_list:
                example_list.append(example_i)
                example_names = example_i
                break

        # split val set out of example and test set
        for n in all_names:
            if n not in example_names and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)

        if args.seg:
            rest_names = []
            for ln in labeled_names:
                if ln not in example_names and ln not in neg_names:
                    rest_names.append(ln)

        if 'TCGA' in args.wsi_path:
            rest_names = check_different_patient(example_names, rest_names, 'TCGA')
        if 'LN' in args.wsi_path:
            rest_names = check_different_patient(example_names, rest_names, 'LN')

        random.shuffle(rest_names)
        val_num = args.val_num if args.val_ratio < 0 else int(len(rest_names) * args.val_ratio)
        val_names = rest_names[:val_num]

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            if args.val_ratio < 0:
                test_names = rest_names[-args.test_num:]
            else:
                test_names = rest_names[val_num:]
            if len(val_names) + len(test_names) > len(rest_names):
                print('wrong split size !!!')
        else: # take partial test slides for tcga cross races
            random.shuffle(test_names)
            if args.test_num > 0:
                test_names = test_names[:args.test_num]

        records['repeat_' + str(i)]['split'] = {'example_names': example_names, 'val_names': val_names, 'test_names': test_names}

        # ====================== run for each class ======================

        # for subtyping, use different example for each cls and apply marco metics
        for cls in range(1, args.c + 1):

            # load example
            example_feats, example_labels = [], []
            pos_feats, neg_feats = [], []

            # ====================== process example ======================

            for n in example_names:
                example_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()

                # empty patch label for image label or sparse label where there is no offline gt
                if args.prompt_type == 'mask':
                    pl = example_n['patch_labels']

                    # binary use 0 normal, 1 tumor, while subtyping use 0 other cls, 1 this cls, 255 normal
                    if args.c > 1:
                        pl[pl == 0] = 255
                        if example_n['wsi_label'] != cls:
                            pl[pl == 1] = 0
                        else:
                            pl[pl == 1] = 1

                else:
                    pl = np.zeros(example_n['features'].shape[0]) - 1

                # load sparse label
                # slideLabel + subtyping is uniqe in pseudo label generation
                if args.prompt_type == "slideLabel" and args.c > 1:
                    if example_n['wsi_label'] != cls:
                        pl[:] = 0
                    else:
                        pl[:] = 1

                # for box, RoughMask and binary + slideLabel, -1 is uncertain pos, 0 is normal
                elif args.prompt_type != 'mask' :
                    pl = load_weak_prompts(n, example_n['wsi_label'], args.wsi_path, pl, \
                        example_n['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale)

                    #  record wsi label for each patch for later label convert
                    if args.c > 1:
                        pl[pl == 0] = 255
                        pl[pl == -1] = 1 if example_n['wsi_label'] == cls else 0
                
                if 'prototype' in mode:
                    pos_feats.append(example_n['features'][(pl != 0) * (pl != 255)])
                    neg_feats.append(example_n['features'][pl == 0])

                if 'knn' in mode:

                    if args.prompt_type != 'slideLabel':
                        feat_fg = example_n['features'][(pl != 0) * (pl != 255)]
                        if feat_fg.shape[0] != 0:
                            if 'mean' in mode:
                                example_feats.append(feat_fg.mean(0, keepdims=True))
                            elif 'max' in mode:
                                example_feats.append(feat_fg.max(0, keepdims=True))
                            example_labels.append(1)
                        
                        feat_bg = example_n['features'][pl == 0]
                        if feat_bg.shape[0] != 0:
                            if 'mean' in mode:
                                example_feats.append(feat_bg.mean(0, keepdims=True))
                            elif 'max' in mode:
                                example_feats.append(feat_bg.max(0, keepdims=True))
                            example_labels.append(0)
                    else:
                        feat = example_n['features']
                        if 'mean' in mode:
                            example_feats.append(feat.mean(0, keepdims=True))
                        elif 'max' in mode:
                            example_feats.append(feat.max(0, keepdims=True))
                        example_labels.append(1 if example_n['wsi_label'] == cls else 0)

            if 'prototype' in mode:
                example_labels = [1, 0]
                pos_feats = np.concatenate(pos_feats, 0)
                neg_feats = np.concatenate(neg_feats, 0)

                if 'simple_shot' in mode:
                    mean_feat = np.concatenate([pos_feats, neg_feats], 0).mean(0)
                    pos_feats -= mean_feat
                    pos_feats = pos_feats.mean(0, keepdims=True)
                    pos_feats = pos_feats / np.linalg.norm(pos_feats, 2, 1, keepdims=True)
                    neg_feats -= mean_feat
                    neg_feats = neg_feats.mean(0, keepdims=True)
                    neg_feats = neg_feats / np.linalg.norm(neg_feats, 2, 1, keepdims=True)
                    example_feats = [pos_feats, neg_feats]
                else:
                    example_feats = [pos_feats.mean(0, keepdims=True), neg_feats.mean(0, keepdims=True)]

            example_feats = torch.tensor(np.concatenate(example_feats, 0)).cuda()
            example_labels = torch.tensor(example_labels).cuda()

            # ====================== inference for test slides ======================

            # predict query
            val_preds, test_preds, val_labels, test_labels = [], [], [], []
            all_query_names = val_names + test_names
            for n in all_query_names:
                query_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                query_feats = torch.tensor(query_n['features']).cuda()
                query_patch_names = query_n['patch_names']
                if args.c > 1:
                    label = query_n['wsi_label'] == cls
                else:
                    label = query_n['wsi_label']

                if 'prototype' in mode:
                    if 'simple_shot' in mode:
                        query_feats -= torch.tensor(mean_feat).cuda()
                        query_feats = query_feats / torch.linalg.norm(query_feats, 2, 1, keepdims=True)

                    topk = min(args.top_instance, query_feats.shape[0])
                    prob = query_feats @ example_feats[0]
                    wsi_pred = prob.topk(topk)[0].mean()

                    if args.vis_path != '' or args.seg:
                        wsi_suffix = os.listdir(args.wsi_path)[0].split('.')[-1]
                        wsi_path = os.path.join(args.wsi_path, n + '.' + wsi_suffix)
                        wsi = openslide.OpenSlide(wsi_path)
                        size = (wsi.level_dimensions[0][1] // args.patch_scale, wsi.level_dimensions[0][0] // args.patch_scale)
                        patch_pred = torch.zeros(size).cuda() + 255
                        idx_in_map = []
                        for pi, pn in enumerate(query_patch_names):
                            x, y = pn.split('/')[-1].split('.')[0].split('_')
                            try:
                                patch_pred[int(y), int(x)] = prob[pi]
                                idx_in_map.append(int(y) * patch_pred.shape[1] + int(x))
                            except:
                                if len(idx_in_map) != 0:
                                    idx_in_map.append(idx_in_map[-1])
                                else:
                                    idx_in_map.append(0)
                                continue

                        if args.vis_path != '' and n in test_names:
                            os.makedirs(args.vis_path, exist_ok=True)
                            np.save(os.path.join(args.vis_path, n + '_' + str(cls) + '.npy'), patch_pred.cpu().numpy())

                        if args.seg:
                            smooth = GaussianBlur(7, 3)
                            fg = patch_pred != 255
                            bg = fg == False
                            smooth_pred = patch_pred.clone()
                            smooth_pred[bg] = smooth_pred[fg].mean() # replace 255 to mean value before smoothing
                            smooth_pred = smooth(smooth_pred.reshape(1, 1, smooth_pred.shape[0], smooth_pred.shape[1]))[0,0]
                            patch_pred[fg] = smooth_pred[fg]
                            patch_pred_list = patch_pred.reshape(-1)[idx_in_map]

                elif 'knn' in mode:
                    if 'mean' in mode:
                        query_feats = query_feats.mean(0)
                    elif 'max' in mode:
                        query_feats = query_feats.max(0)[0]
                    else:
                        print('false eval mode')
                   
                    pos_example_feats, neg_example_feats = example_feats[example_labels == 1], example_feats[example_labels == 0]
                    wsi_pred = (pos_example_feats @ query_feats).topk(min(5, pos_example_feats.shape[0]))[0].mean() - \
                            (neg_example_feats @ query_feats).topk(min(5, neg_example_feats.shape[0]))[0].mean()
                    
                else:
                    print('false eval mode')

                if args.seg:
                    pred = torch.tensor(patch_pred_list)
                    label = torch.tensor(query_n['patch_labels'])
                else:
                    pred = torch.tensor([wsi_pred])
                    label = torch.tensor([label])

                if n in val_names:
                    val_preds.append(pred)
                    val_labels.append(label)
                else:
                    test_preds.append(pred)
                    test_labels.append(label)

            # ====================== process validation set and assign label ======================

            # search a threshold to predict label on val set for fair comparisions
            val_preds = torch.cat(val_preds).cpu()
            val_labels = torch.cat(val_labels)
            val_auc = aucroc(val_preds, val_labels).item()
            test_preds = torch.cat(test_preds).cpu()
            test_labels = torch.cat(test_labels)
            
            precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
            accs = np.array([((val_preds > _).float() == val_labels).sum() / val_labels.shape[0] for _ in thresholds])
            if args.seg:
                f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_acc_score = accs[best_f1_score_index]
                thresh = thresholds[best_f1_score_index]
            else:
                best_acc_score = np.max(accs[np.isfinite(accs)])
                best_acc_score_index = np.argmax(accs[np.isfinite(accs)])
                thresh = thresholds[best_acc_score_index]

            preds = test_preds
            thresh_preds = (test_preds > thresh).float()
            labels = test_labels
            acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
            rec = ((thresh_preds * labels).sum() / labels.sum()).cpu().item()
            pre = ((thresh_preds * labels).sum() / thresh_preds.sum()).cpu().item()
            auc = aucroc(preds, labels).item()
            f1 = rec * pre * 2 / (rec + pre) if rec + pre > 0 else 0

            auc_list.append(auc)
            f1_list.append(f1)
            acc_list.append(acc)

            s = 'class:' + str(cls) + ' val auc:' + str(round(val_auc, 4)) + ', test auc:' + str(round(auc, 4)) + ', val acc: ' \
                 + str(round(best_acc_score, 4)) + ', test f1: ' + str(round(f1, 4)) + ', test acc: ' + str(round(acc, 4))
            print(s)
            txt_rec.append(s)
            records['repeat_' + str(i)]['results_cls' + str(cls)] = {'val_auc': round(val_auc, 4), 'test_auc': round(auc, 4), \
                    'val_acc': round(best_acc_score, 4), 'test_f1': round(f1, 4), 'test_acc': round(acc, 4)}
            records['repeat_' + str(i)]['pred_cls' + str(cls)] = {'labels': labels.cpu().tolist(), \
                    'logits': preds.cpu().tolist(), 'preds': thresh_preds.cpu().tolist()}

    # ====================== count and record results ======================

    auc_mean = np.array(auc_list).mean()
    macro_auc = macro_value(auc_list, args.c)
    auc_std = np.array(macro_auc).std()
    f1_mean = np.array(f1_list).mean()
    macro_f1 = macro_value(f1_list, args.c)
    f1_std = np.array(macro_f1).std()
    acc_mean = np.array(acc_list).mean()
    macro_acc = macro_value(acc_list, args.c)
    acc_std = np.array(macro_acc).std()

    s = 'auc mean: ' + str(round(auc_mean, 4)) + ', auc std: ' + str(round(auc_std, 4)) + \
        ', f1 mean: ' + str(round(f1_mean, 4)) + ', f1 std: ' + str(round(f1_std, 4)) + \
        ', acc mean: ' + str(round(acc_mean, 4)) + ', acc std: ' + str(round(acc_std, 4))
    print(s)
    txt_rec.append(s)

    records['mean'] = {'auc_mean': round(auc_mean, 4), 'auc_std': round(auc_std, 4), 'auc_values': macro_auc, \
            'f1_mean': round(f1_mean, 4), 'f1_std': round(f1_std, 4), 'f1_values': macro_f1, \
            'acc_mean': round(acc_mean, 4), 'acc_std': round(acc_std, 4), 'acc_values': macro_acc}
    records['text_records'] = txt_rec

    return records


# ====================== the main function ======================

if __name__ == '__main__':

    # ====================== arg parser ======================

    parser = argparse.ArgumentParser('Multiple Instance Prompting')
    parser.add_argument('--mode', default='search', type=str, help="update: update features, inference: process query only, \
        eval: load processed features for evaluate, default: update and test")

    # hyper-params
    parser.add_argument('--topk', default=40, type=int, help='Number of top patchs to take')
    parser.add_argument('--top_instance', default=1, type=int, help='Number of top patchs to take')
    parser.add_argument('--temperature', default=10, type=float, help='Temperature for sample reweights')
    parser.add_argument('--related_thresh', default=0.88, type=float, help='cosine similarity threshold to select related patchs')
    parser.add_argument('--example_num', default=3, type=int, help='number of wsi for init example')
    parser.add_argument('--multiple_num', type=int, nargs='+', default=None, help='multi example num')

    # dataset information and settings
    parser.add_argument('--raw_feature_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--wsi_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--dump_features', default=None, help='Path where to save features')
    parser.add_argument('--dump_pseudo', default='', help='Path where to save pseudo, vis and data split')
    parser.add_argument('--dump_records', default='', help='Path to save records (json file)')
    parser.add_argument('--vis_path', default='', help='Path where to save heatmap')
    parser.add_argument('--dataset_info', default='/path/to/data_list_gt_and_split', type=str, help='json file recording dataset info')
    parser.add_argument('--patch_scale', default=512, type=int, help='patch size in 40x for anno loading')
    parser.add_argument('--file_min_size', default=5000, type=int, help='skip background and patches with a few content')
    parser.add_argument('--c', default=1, type=int, help='number of class, c >1 for subtyping')
    parser.add_argument('--seg', default=False, action='store_true', help='True to evaluate segmentation task (f1 = dice)')

    # for weak prompts
    parser.add_argument('--prompt_type', default='mask', help='prompttation type')
    parser.add_argument('--prompt_path', default='', help='path of prompttation xml file')
    parser.add_argument('--ignore', default=0, type=float, help='degree to ignore uncertain example (during generating example)')
    parser.add_argument('--ignore_query', default=0.3, type=float, help='degree to ignore uncertain foreground query (subtyping only)')

    # test settings
    parser.add_argument('--seed', default=1024, type=int, help='for the reproduce of data split')
    parser.add_argument('--runs', default=5, type=int, help='number of test times')
    parser.add_argument('--val_num', default=100, type=int, help='number of validation WSIs')
    parser.add_argument('--test_num', default=129, type=int, help='number of test WSIs')
    parser.add_argument('--val_ratio', default=-1, type=float, help='split val test via ratio to replace specific number')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.dump_features, exist_ok=True)

    # collect features and information
    feature_processor(args)

    # ====================== Execute different modes ======================
    
    # evaluat with given hyper-parameters (in deployment)
    if args.mode == 'eval':
        print(args)
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            random.seed(args.seed)
            args.example_num = p
            res, rec = evaluate(args)
            records[str(p) + '-shot'] = rec

        if args.dump_records != '':
            np.save(args.dump_records, records)
    
    # run baselines
    if args.mode == 'baselines':
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            args.example_num = p
            records[str(p) + '-shot'] = {}

            # segmentation need patch predictions, knn is conducted on wsi-level
            if not args.seg and args.vis_path == '':
                print('mode: knn_mean, example ' + str(args.example_num))
                random.seed(args.seed)
                rec_knn_mean = evaluate_baseline(args, 'knn_mean')
                records[str(p) + '-shot']['knn_mean'] = rec_knn_mean
            
                print('mode: knn_max, example ' + str(args.example_num))
                random.seed(args.seed)
                rec_knn_max = evaluate_baseline(args, 'knn_max')
                records[str(p) + '-shot']['knn_max'] = rec_knn_max
            
            print('mode: prototype, example ' + str(args.example_num))
            random.seed(args.seed)
            rec_proto = evaluate_baseline(args, 'prototype')
            records[str(p) + '-shot']['prototype'] = rec_proto

            print('mode: prototype_simple_shot, example ' + str(args.example_num))
            random.seed(args.seed)
            rec_simp = evaluate_baseline(args, 'prototype_simple_shot')
            records[str(p) + '-shot']['simple_Shot'] = rec_simp

        if args.dump_records != '':
            np.save(args.dump_records, records)

    # run in val-test set with hyperparameter search
    if args.mode == 'default': 

        pseudo = args.dump_pseudo
        args.dump_pseudo = ''

        # speed up param search
        if args.c > 1:
            ori_runs = args.runs
            ori_val_num = args.val_num
            args.runs=3
            args.val_num=50
        
        # search for parameters (in extended data figure 10)
        if args.c > 1:
            v, t = 0, 0
            for p in [1000, 2000, 3000, 4000, 5000]:
                print('searching top_instance, param: ' + str(p))
                random.seed(args.seed) # validate params without influence from sampling
                args.top_instance = p
                res, _ = evaluate(args, val_only=True)
                if res > v:
                    v = res
                    t = p
            args.top_instance = t
            print('params: top_instance, searched threshold: ' + str(t) + ', mean:' + str(v))

        v, t = 0, 0
        for p in [0, 0.02, 0.04, 0.06, 0.08]:
            print('searching ignore, param: ' + str(p))
            random.seed(args.seed)
            args.ignore = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.ignore = t
        print('params: ignore, searched threshold: ' + str(t) + ', mean:' + str(v))

        if args.c > 1:
            v, t = 0, 0
            for p in [0.1, 0.15, 0.2, 0.25, 0.3]:
                print('searching ignore-query, param: ' + str(p))
                random.seed(args.seed)
                args.ignore_query = p
                res, _ = evaluate(args, val_only=True)
                if res > v:
                    v = res
                    t = p
            args.ignore_query = t
            print('params: ignore-query, searched threshold: ' + str(t) + ', mean:' + str(v))

        v, t = 0, 0
        for p in [20, 30, 40, 50, 60]:
            print('searching topk, param: ' + str(p))
            random.seed(args.seed)
            args.topk = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.topk = t
        print('params: topk, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [0.86, 0.87, 0.88, 0.89, 0.9]:
            print('searching related_thresh, param: ' + str(p))
            random.seed(args.seed)
            args.related_thresh = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.related_thresh = t
        print('params: related_thresh, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [1, 5, 10, 20, 30]:
            print('searching temperature, param: ' + str(p))
            random.seed(args.seed)
            args.temperature = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.temperature = t
        print('params: temperature, searched threshold: ' + str(t) + ', mean:' + str(v))
        
        # eval with searched params and test influence of example number
        if args.c > 1:
            args.runs = ori_runs
            args.val_num = ori_val_num
        args.dump_pseudo = pseudo

        print(args)
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            random.seed(args.seed)
            args.example_num = p
            res, rec = evaluate(args)
            records[str(p) + '-shot'] = rec
        
        # save results
        if args.dump_records != '':
            np.save(args.dump_records, records)
