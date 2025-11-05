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
import numpy as np
import utils

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def extract_feature_pipeline(args):

    # ====================== building network ======================

    transform = None

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

    elif "res50" in args.arch: # non pathological foundation model
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)

    # load uni model from local weights, please download it in advance
    elif args.arch == 'uni':
        import timm
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        utils.load_pretrained_weights(model, args.pretrained_weights)

    # load from hugging face, you are required to clone the code in the home dir with its dependencies
    elif args.arch == 'plip':
        sys.path.append(os.path.expanduser('~/plip'))
        from plip import PLIP
        plip_model = PLIP('vinid/plip')
        model = plip_model.model

    # load from hugging face
    elif args.arch == 'gigapath':
        import timm
        from huggingface_hub import login
        login("xxxxxxx") # write your access token here
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    # load from hugging face, you are required to clone the code in the home dir with its dependencies
    elif args.arch == 'chief':
        sys.path.append(os.path.expanduser('~/CHIEF'))
        from models.ctran import ctranspath
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'ckpt/CHIEF_CTransPath.pth')
        model.load_state_dict(td['model'], strict=True)

    # load from hugging face. Note that you need to crop the images in 512 instead of defualt 256
    elif args.arch == 'titan':
        from huggingface_hub import login
        from transformers import AutoModel
        login("xxxxxxx") # write your access token here
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, transform = titan.return_conch()

    # load from hugging face
    elif args.arch == 'virchow':
        from huggingface_hub import login
        login("xxxxxxx") # write your access token here
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    # load from hugging face, you are required to clone the code in the home dir with its dependencies, img crop to 384
    elif args.arch == 'musk':
        sys.path.append(os.path.expanduser('~/MUSK/musk'))
        import modeling
        import musk_utils
        from timm.models import create_model
        import torchvision
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        model = create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
        model.to(device="cuda", dtype=torch.float16)
        img_size = 384
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    # load from hugging face, you are required to clone the code in the home dir with its dependencies
    elif args.arch == 'gpfm':
        sys.path.append(os.path.expanduser('~/GPFM'))
        from models import get_model
        model = get_model('GPFM', 0, 1)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)

    model.cuda()
    model.eval()

    # ====================== preparing dataset ======================

    # load dataset after network to avoid torchvision conflict from huggingface remote_code
    from dataset.wsi_dataset import pth_transforms, AutoPad, ImagePyramidDataset

    # uncommant it and RandomDistortions() in transform to use camelyon16c
    #from dataset.camelyon16c import RandomDistortions

    # apply 5-crop for all models as decribed in the paper
    if transform == None:
        transform = pth_transforms.Compose([
            AutoPad(256),
            pth_transforms.FiveCrop(224),
            #RandomDistortions()
            pth_transforms.Lambda(lambda crops: torch.stack([pth_transforms.ToTensor()(crop) for crop in crops])),
            pth_transforms.Lambda(lambda crops: torch.stack([pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops])),
        ])
    dataset = ImagePyramidDataset(args.data_path, [[0, 'x20']], transform, args.dump_features)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )


    print("Extracting features ...")
    extract_features(model, data_loader, args.use_cuda, args.dump_features, args.arch)


# ====================== extract features =====================

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, dump_features='', arch=''):
    metric_logger = utils.MetricLogger(delimiter="  ")
    for imgs, out_dirs, out_paths in metric_logger.log_every(data_loader, 10):
        imgs = imgs.cuda(non_blocking=True)

        # size in 224 use 5-crop while others use their raw transform
        if len(imgs.size()) == 4:
            bs, c, h, w = imgs.size()
            nc = 1
        else:
            bs, nc, c, h, w = imgs.size()
            imgs = imgs.reshape(-1, c, h, w)

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
        
        feats = feats.reshape(bs, nc, -1).mean(1).cpu().numpy()
        
        # save the patch features
        for i in range(len(out_dirs)):
            od = os.path.join(dump_features, out_dirs[i])
            os.makedirs(od, exist_ok=True)
            np.save(os.path.join(dump_features, out_paths[i]), feats[i])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract feature for image pyramid')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    extract_feature_pipeline(args)

    dist.barrier()
