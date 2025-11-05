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

import os, math, copy, math
import numpy as np
import openslide
import torch
import cv2


# ====================== prompt loader ======================

def load_weak_prompts(fn, wsi_label, wsi_dir, patch_labels, patch_names, anno_dir, anno_type, side=512):

    # img label assign 0 for all neg
    if anno_type == "slideLabel":
        if wsi_label == 0:
            patch_labels[:] = 0
        else:
            patch_labels[:] = -1
    
    else:
        # load positions
        pos = []
        for pn in patch_names:
            x, y = pn.split('/')[-1].split('.')[0].split('_')
            x, y = int(x), int(y)
            pos.append([y, x])
        pos = np.array(pos)
   
        # load xml anno
        s = open(os.path.join(anno_dir, fn + '.xml')).read()
        tks = s.split('<Annotation Id="')[1:]
     
        if anno_type == "box":
            boxes = []
            for tk in tks:
                if tk[0] == '2':
                    for b in tk.split('<Region Id=')[1:]:
                        ps = b.split('<Vertex X="')
                        x_list, y_list = [], []
                        for p in ps[1:]:
                            x_list.append(int(float(p.split('"')[0]) / side + 0.5))
                            y_list.append(int(float(p.split('"')[2]) / side + 0.5))
                        x1, x2 = min(x_list), max(x_list)
                        y1, y2 = min(y_list), max(y_list)
                        boxes.append([x1, y1, x2, y2])
        
            for i in range(pos.shape[0]):
                p = pos[i]
                in_box = False
                for b in boxes:
                    if p[0] >= b[1] and p[0] <= b[3] and p[1] >= b[0] and p[1] <= b[2]:
                        in_box = True
                if not in_box:
                    patch_labels[i] = 0

        elif anno_type == 'roughMask':
            slide = openslide.OpenSlide(os.path.join(wsi_dir, fn + '.svs'))
            w, h = slide.level_dimensions[0]
            if (w % side) != 0 or (h % side) != 0:
                w += side - w % side
                h += side - h % side

            mid_scale = side // 32 # keep anno details, original img is too large to process
            resize_scale = side // mid_scale
            out = np.zeros((h // mid_scale, w // mid_scale, 3)).astype('uint8') # patch level gt

            roi_contours = []
            for tk in tks:
                if tk[0] == '3':
                    for roi in tk.split('<Region Id="')[1:]:
                        points = []
                        for p in roi.split(' X="')[1:]:
                            _ = p.split('" Y="')
                            points.append([int(float(_[0]) / mid_scale + 0.5), int(float(_[1].split('"')[0]) / mid_scale + 0.5)])
                        roi_contours.append(np.array(points))
            
            # resize by keep max value (label=1 if 1 in the window)
            out = cv2.fillPoly(out, roi_contours, [0, 0, 1])
            out = out[:, :, -1]
            out = out.reshape(out.shape[0] // resize_scale, resize_scale, out.shape[1] // resize_scale, resize_scale)
            out = out.max(1)
            out = out.max(2)

            for i in range(pos.shape[0]):
                if out[pos[i][0], pos[i][1]] == 0:
                    patch_labels[i] = 0
             
        else :
             print('wrong anno_type')
              
    return patch_labels
             

# ====================== some util functions ======================

def compute_similarity(query, example, topk=40):
    sim = low_memory_matrix_multiply(example, query.t())
    if topk > 0:
        sim, _ = topk_low_memory_(sim, min(topk, sim.shape[0]))
        sim = sim.mean(0)
    else:
        sim = sim.mean(0)

    return sim


def low_memory_matrix_multiply(A, B, max_size=20000):
    na,  nb = A.shape[0], B.shape[-1]
    sim = torch.FloatTensor(na, nb)

    for i in range(math.ceil(na / max_size)):
        for j in range(math.ceil(nb / max_size)):
            i1, i2 = i * max_size, (i + 1) * max_size
            j1, j2 = j * max_size, (j + 1) * max_size
            sim[i1: i2, j1: j2] = (A[i1: i2, :] @ B[:, j1: j2]).cpu()

    return sim


def topk_low_memory(inp, n, dim):
    scores, idxs = [], []
    offset = 0
    chunk_num = max(1, inp.shape[dim] // 10000)
    for i in inp.chunk(chunk_num, dim):
        score, idx = i.cuda().topk(min(n, i.shape[dim]), dim)
        scores.append(score)
        idxs.append(idx + offset)
        offset += i.shape[dim]

    scores = torch.cat(scores, dim)
    scores, idxs2 = scores.topk(min(n, scores.shape[dim]), dim)
    idxs = torch.cat(idxs, dim).transpose(0, dim) # ori index
    idxs2 = idxs2.transpose(0, dim) # idx after reduction
    out_idxs = []
    for i in range(idxs2.shape[-1]):
        out_idxs.append(idxs[idxs2[:, i], i])
    out_idxs = torch.stack(out_idxs, -1).transpose(0, dim)

    return scores, out_idxs


# large query number lead to "CUDA error: an illegal memory access was encountered"
# rare WSI (patch > 20000) topk == topk * math.ceil(num / 20000)
def topk_low_memory_(inp, n):
    scores, idxs = [], []
    for i in range(math.ceil(inp.shape[1] / 20000)):
        a, b = topk_low_memory(inp[:, i * 20000: i * 20000 + 20000], n, 0)
        scores.append(a)
        idxs.append(b)

    return torch.cat(scores, 1), torch.cat(idxs, 1)


# ====================== tagging visualization ======================

# vis_dir visualizes heatmap and pseudo_label with ori image, mask_dir saves the pseudo_label
def vis_heat(score, label, pos, f, wsi_dir, vis_dir, mask_dir, side=512):
    f = f if '.svs' in f else f.replace('.xml', '.svs')
    if not os.path.exists(os.path.join(wsi_dir, f)):
        f = f.replace('.svs', '.tif')
    wsi = openslide.OpenSlide(os.path.join(wsi_dir, f))
    w, h = wsi.level_dimensions[0]
    w = math.ceil(w / 512)
    h = math.ceil(h / 512)
    heat = np.zeros((h, w))
    label_map = np.zeros((h, w)) + 253 # 255 normal, 254 uncertain, 253 defualt, subtyping (0 neg, 1 pos), binary(0 noraml, 1 pos)

    for i in range(score.shape[0]):
        y, x = pos[i]
        heat[y, x] = score[i]
        lb = label[i]
        if lb == -1:
            lb = 254
        label_map[y, x] = lb

    # only vis wsi with low resolution to avoid out of memory
    if len(wsi.level_dimensions) > 1:
        scale = len(wsi.level_dimensions) - 1 #3
        img = np.array(wsi.read_region((0, 0), scale, wsi.level_dimensions[scale]))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fg = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC) > 0
        heat = cv2.applyColorMap((heat * 255).astype('uint8'), cv2.COLORMAP_JET) * 0.4
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        vis_heat = img.copy()
        vis_heat[fg] = img[fg] * 0.6 + heat[fg]

        vis_label = copy.deepcopy(label_map)
        if 255 in vis_label: # subtyping
            vis_label[vis_label == 255] = -1 # normal to temp label
            vis_label[(vis_label < 253) * (vis_label >= 0)] = 255 # all pos (0 ,1, ...)
            vis_label[vis_label == -1] = 0
            vis_label[vis_label == 253] = 0 # default
            vis_label[vis_label == 254] = 128 # uncertain
        else:
            vis_label[vis_label == 253] = 0
            vis_label[vis_label == 254] = 128
            vis_label[vis_label == 1] = 255
        vis_label = vis_label.astype('uint8')
        vis_label = cv2.applyColorMap((vis_label), cv2.COLORMAP_JET) * 0.4
        vis_label = cv2.resize(vis_label, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img[fg] = img[fg] * 0.6 + vis_label[fg]
        vis_label = img

        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(vis_dir, f.split('.')[0] + '_score.jpg'), vis_heat)
        cv2.imwrite(os.path.join(vis_dir, f.split('.')[0] + '_label.jpg'), vis_label)

    label_map = label_map.astype('uint8')
    cv2.imwrite(os.path.join(mask_dir, f.split('.')[0] + '.png'), label_map)


# ====================== instance miner for subtyping ======================

def execute_miner(neg_example_feats, feats, names, topk=40, uncertain=0.2):
    sim = []
    for i in range(math.ceil(neg_example_feats.shape[0] / 10000)):
        sim.append((neg_example_feats[i * 10000: (i + 1) * 10000] @ feats.t()).cpu())
    sim = torch.cat(sim, 0)#.cuda()
    #sim = neg_example_feats @ feats.t()

    sim, _ = topk_low_memory_(sim, min(topk, sim.shape[0]))
    sim = sim.mean(0)
    ukn = torch.ones(len(names)) == 1
    fg = torch.zeros(len(names)).long()
    fg = basic_tagger(sim, ukn, fg, uncertain, positive=False)
    fg = fg == 1
    
    if fg.sum() == 0:
        return feats, names

    out_names = []
    for i in fg.nonzero()[:, 0].cpu().numpy():
        out_names.append(names[i])
    return feats[fg], out_names


# ====================== basic tagger (algorithm 1) ======================

def basic_tagger(ukn_sim, ukn_mask, label, uncertain, positive=False):
    thresh, _ = cv2.threshold((ukn_sim.cpu().numpy() * 255).astype('uint8'), \
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = (ukn_sim.max() - ukn_sim.min()) * uncertain
    low_t = float(thresh) / 255 - diff
    high_t = float(thresh) / 255 + diff

    ukn_label = label[ukn_mask]
    if positive:
        ukn_label[ukn_sim > low_t] = 1
        ukn_label[ukn_sim < high_t] = 0
    else:
        ukn_label[ukn_sim < low_t] = 1
        ukn_label[ukn_sim > high_t] = 0
    ukn_label[(ukn_sim <= high_t) * (ukn_sim >= low_t)] = -1
    label[ukn_mask] = ukn_label

    return label


# ====================== in-context tagger (algorithm 2) ======================

# binary classification via sparse annotations (slideLabel, box, roughMask)
def execute_tagger(feats, labels, patch_names, wsi_names, \
        vis_info=None, uncertain=0.1, topk=40, sampling_size=-1):

    # assign init label for each wsi
    # record uncertain positive and normal patch idx
    info_dic = {}
    pos_idx, neg_idx = [], []
    for n in wsi_names:
        pos, idx = [], []
        for i, pn in enumerate(patch_names):
            if n in pn:
                x, y = pn.split('/')[-1].split('.')[0].split('_')
                pos.append([int(y), int(x)])
                idx.append(i)
        pos = np.array(pos)
        info_dic[n] = {'idx': idx, 'pos': pos}

        ukn_mask = labels[idx] == -1
        if True in ukn_mask:
            sim_ukn = compute_similarity(feats[idx][ukn_mask], feats[labels == 0], topk=topk)
            sim_ukn = (sim_ukn - sim_ukn.min()) / (sim_ukn.max() - sim_ukn.min())
            labels_ukn = torch.zeros(sim_ukn.shape[0]).cuda().long()
            labels_ukn = basic_tagger(sim_ukn, labels_ukn==0, labels_ukn, uncertain, positive=False)
            idx_ukn = [idx[_] for _ in ukn_mask.nonzero()[:, 0]]
            pos_idx.extend([idx_ukn[_] for _ in (labels_ukn == 1).nonzero()[:, 0]])
            neg_idx.extend([idx_ukn[_] for _ in (labels_ukn == 0).nonzero()[:, 0]])
            neg_idx.extend([idx[_] for _ in (labels[idx] == 0).nonzero()[:, 0]])
        else:
            neg_idx.extend(idx)

    # assign label via dataset-level pos and neg
    for n in wsi_names:
        idx_n = info_dic[n]['idx']
        if -1 in labels[idx_n]:
            feats_n = feats[idx_n]
            sim_pos = compute_similarity(feats_n, feats[pos_idx], topk=topk)
            sim_neg = compute_similarity(feats_n, feats[neg_idx], topk=topk)
            score = sim_pos - sim_neg
            score = torch.clamp(score, -0.5, 0.5) + 0.5 # norm to 0-1 for vis via clamp

            labels_n = torch.zeros(feats_n.shape[0]).cuda().long()
            labels_n = basic_tagger(score, labels_n==0, labels_n, uncertain, True)
            labels[idx_n] = labels_n

            if vis_info != None:
                vis_heat(score, labels_n, info_dic[n]['pos'], n + '.svs', vis_info['wsi_dir'], \
                    vis_info['vis_dir'], vis_info['mask_dir'], side=512)
    
    return labels


# ====================== in-context tagger for subtyping (algorithm 2) ======================

# 1.dataset similarity 2.init pseudo in each wsi 3.refine via dataset pseduo
def execute_subtyping_tagger(feats, labels, patch_names, wsi_names, \
        vis_info=None, uncertain=0.1, topk=40, sampling_size=40000):
    
    # step1 similarity cross wsi-level label
    pos, neg = labels == 1, labels == 0
    if sampling_size > 0:
        sampled_idx = torch.randint(0, feats[neg].shape[0], (1, sampling_size))[0]
        sim_pos = compute_similarity(feats[pos], feats[neg][sampled_idx[:sampling_size], :], topk=topk)
        sampled_idx = torch.randint(0, feats[pos].shape[0], (1, sampling_size))[0]
        sim_neg = compute_similarity(feats[neg], feats[pos][sampled_idx[:sampling_size], :], topk=topk)
    else:
        sim_pos = compute_similarity(feats[pos], feats[neg], topk=topk)
        sim_neg = compute_similarity(feats[neg], feats[pos], topk=topk)
    sim = torch.zeros(feats.shape[0]).cuda()
    sim[pos] = sim_pos.cuda()
    sim[neg] = sim_neg.cuda()

    # step2 assign init label for each wsi
    # record certain pos and normal patches
    info_dic = {}
    labeled_idx_dic = {255: [], 0: [], 1: []}
    for n in wsi_names:
        pos, idx = [], []
        for i, pn in enumerate(patch_names):
            if n in pn:
                x, y = pn.split('/')[-1].split('.')[0].split('_')
                pos.append([int(y), int(x)])
                idx.append(i)
        pos = np.array(pos)
        info_dic[n] = {'idx': idx, 'pos': pos}

        sim_n = sim[idx]
        sim_n = (sim_n - sim_n.min()) / (sim_n.max() - sim_n.min())
        labels_n = torch.zeros(sim_n.shape[0]).cuda().long()
        labels_n = basic_tagger(sim_n, labels_n ==0, labels_n, uncertain, positive=False)

        wsi_label = labels[idx[0]].item()
        labeled_idx_dic[wsi_label].extend([idx[_] for _ in (labels_n == 1).nonzero()[:, 0]])
        labeled_idx_dic[255].extend([idx[_] for _ in (labels_n == 0).nonzero()[:, 0]])

    # step3 assign label via dataset pos and neg
    for n in wsi_names:
        idx_n = info_dic[n]['idx']
        feats_n = feats[idx_n]
        wsi_label = labels[idx_n[0]].item()
        sim_pos = compute_similarity(feats_n, feats[labeled_idx_dic[wsi_label]], topk=-1)
        sim_neg = compute_similarity(feats_n, feats[labeled_idx_dic[255]], topk=-1)
        score = sim_pos - sim_neg
        score = torch.clamp(score, -0.5, 0.5) + 0.5 # norm to 0-1 for vis via clamp

        labels_n = torch.zeros(feats_n.shape[0]).cuda().long()
        labels_n = basic_tagger(score, labels_n==0, labels_n, uncertain, True)
        labels_n[labels_n == 0] = 255 # normal cells
        labels_n[labels_n == -1] = 254 # uncertain
        labels_n[labels_n == 1] = wsi_label
        labels[idx_n] = labels_n

        if vis_info != None:
            vis_heat(score, labels_n, info_dic[n]['pos'], n + '.svs', vis_info['wsi_dir'], \
                vis_info['vis_dir'], vis_info['mask_dir'], side=512)

    return labels


# ====================== inference with classifier, aggregator, post processor ======================

def inference(args, example_feats, example_labels, example_patch_names,
    query_feats, query_patch_names, wsi_size, top_instance=1, vis_info=None, smooth=None):

    # ====================== in-context classifier ======================

    # we name a test case a query to avoid confusin confusion with test set
    query_feats = query_feats.t()

    # aviod out of memory to split matrix and concat in cpu
    cosine = []
    for i in range(math.ceil(example_feats.shape[0] / 10000)):
        cosine.append((example_feats[i * 10000: (i + 1) * 10000] @ query_feats).cpu())
    cosine = torch.cat(cosine, 0)#.cuda()
    example_labels = example_labels.cpu()

    pos_cosine = cosine[example_labels == 1]
    pos_cosine, pos_example_idxs = topk_low_memory_(pos_cosine, args.topk)
    neg_cosine = cosine[example_labels == 0]
    neg_cosine, neg_example_idxs = topk_low_memory_(neg_cosine, args.topk)

    query_logits = pos_cosine.cuda().mean(0) - neg_cosine.cuda().mean(0)

    # ====================== attention aggregator ======================

    # using related patchs and topk
    top_query_num = min(top_instance, len(query_patch_names))
    top_query_logits, top_query_idxs = query_logits.topk(top_query_num)

    # self-attention for test patches with high patch score
    wsi_pred_list = []
    for i in range(top_query_num):
        wsi_pred, query_idx_i = top_query_logits[i], top_query_idxs[i]
        sim = query_feats[:, query_idx_i: query_idx_i + 1].t() @ query_feats
        sim_score, sim_idxs = sim[0].sort()
        num = (sim_score > args.related_thresh).sum()
        related_preds = query_logits[sim_idxs[-int(num):]]
        w = (sim_score[-int(num):] * args.temperature).softmax(0)
        wsi_pred = (w * related_preds).sum()
        wsi_pred_list.append(wsi_pred)
    wsi_pred = sum(wsi_pred_list) / len(wsi_pred_list)

    # ====================== patch reshape to wsi for heatmap / seg ======================

    idx_in_map = []
    if wsi_size != None:
        #patch_pred = torch.zeros(wsi_size).cuda() + query_logits.min()
        patch_pred = torch.zeros(wsi_size).cuda() + 255
        patch_pred_list = []
        for i, n in enumerate(query_patch_names):
            patch_pred_list.append(query_logits[i])
            x, y = n.split('/')[-1].split('.')[0].split('_')
            try:
                patch_pred[int(y), int(x)] = query_logits[i]
                idx_in_map.append(int(y) * patch_pred.shape[1] + int(x))
            except:
                if len(idx_in_map) != 0:
                    idx_in_map.append(idx_in_map[-1])
                else:
                    idx_in_map.append(0)
                continue
    else:
        patch_pred, patch_pred_list = None, None

    # ====================== segmentation post processor ======================

    # patch_pred ing
    if smooth != None:
        fg = patch_pred != 255
        bg = fg == False
        smooth_pred = patch_pred.clone()
        smooth_pred[bg] = smooth_pred[fg].mean() # replace 255 to mean value before smoothing
        smooth_pred = smooth(smooth_pred.reshape(1, 1, smooth_pred.shape[0], smooth_pred.shape[1]))[0,0]
        patch_pred[fg] = smooth_pred[fg]
        patch_pred_list = patch_pred.reshape(-1)[idx_in_map]

    return wsi_pred, patch_pred, patch_pred_list
