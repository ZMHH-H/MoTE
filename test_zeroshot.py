import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
import clip
import numpy as np

import yaml
from dotmap import DotMap
from datasets.video import Video_dataset
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from modules.video_clip import video_header, VideoCLIP
from modules.text_prompt import text_prompt, text_prompt_ensemble


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    args = parser.parse_args()
    return args

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # get fp16 model and weight
    model_clip, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st= config.network.joint_st) # Must set jit=False for training  ViT-B/32

    if args.precision == "amp" or args.precision == "fp32":
        model_clip = model_clip.float()


    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    # rescale size
    scale_size = int(config.data.input_size)

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1: # one center crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops (upper left, upper right, lower right, lower left, center)
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10: # 5 normal crops + 5 flipped crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))


    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]),
        dense_sample=args.dense,
        test_clips=args.test_clips)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)


    # ============= generate class features ==============
    print('============= Start encoding class features ===========')
    classes = text_prompt_ensemble(val_data)
    n_class = classes[0].size(0)
    model_clip.cuda()
    model_clip.eval()
    with torch.no_grad():
        # @zmhh_h multi text prompts
        cls_feature_list = [model_clip.encode_text(classes[i].cuda(), return_token=True)[0] for i in range(len(classes))]
        for cls_feature in cls_feature_list:
            cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
        cls_feature = torch.stack(cls_feature_list, 0).mean(0)
        cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    print('============= End encoding class features ===========')

    # CLIP image encoder
    model = VideoCLIP(model_clip, config.data.num_segments)
    del model_clip

    # Temporal Aggregation Module
    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
        config.network.temporal_layer,
        config.network.num_experts)

    # =============== patch clip weights with a ratio of alpha===================
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        checkpoint_patch = {}
        alpha = 0.99
        for k, v in checkpoint['model_state_dict'].items():
            if k in clip_state_dict.keys():
                checkpoint_patch[k]= alpha*v+(1-alpha)*clip_state_dict[k]
            else:
                print('unmatched parameters: ',k)
        if dist.get_rank() == 0:
            print('load model: epoch {}'.format(checkpoint['epoch']))
        # model.load_state_dict(update_dict(checkpoint['model_state_dict']))
        model.load_state_dict(checkpoint_patch)
        video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
        del checkpoint,checkpoint_patch

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        if config.network.sim_header != "None":
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])


    prec1 = validate(
        val_loader, device,
        model, video_head, config, cls_feature, args.test_crops, args.test_clips)
    return


def validate(val_loader, device, model, video_head, config, text_features, test_crops, test_clips):

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    video_head.eval()
    proc_start_time = time.time()

    sim_logits = []     # 
    labels = []     # 
    t_features = []

    with torch.no_grad():
        n_class = text_features.size(0)
        
        for i, (image, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops
            num_crop *= test_clips  # 4 clips for testing when using dense sample

            class_id = class_id.to(device) # [BS]
            n_seg = config.data.num_segments
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()

            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.module.encode_image(image_input)
            tem_features_multiview = video_head.module.agg_video_feat(image_features, text_features)
            tem_features_multiview = tem_features_multiview.reshape(batch_size,num_crop,t,-1).mean(2)
            
            cnt_time = time.time() - proc_start_time
            similarity = video_head(image_features, text_features)
            similarity = F.softmax(similarity, -1)
            similarity = similarity.reshape(batch_size, num_crop, -1).mean(1)
            similarity = similarity.view(batch_size, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)

            ########## gathering 
            t_features.append(concat_all_gather(tem_features_multiview))
            sim_logits.append(concat_all_gather(similarity))
            labels.append(concat_all_gather(class_id))
            ##########


            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i+1) / (batch_size * dist.get_world_size())
                print(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), runtime=runtime, top1=top1, top5=top5)))
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        ## half-classes evaluation
        sim, la = sim_logits[0], labels[0]
        temporal_feat = t_features[0]
        for i in range(1, len(sim_logits)): 
            sim = torch.cat((sim, sim_logits[i]), 0)
            la = torch.cat((la, labels[i]), 0)
            temporal_feat = torch.cat((temporal_feat, t_features[i]), 0)

        text_feat = text_features/ text_features.norm(dim=-1, keepdim=True)
        temporal_feat = temporal_feat/ temporal_feat.norm(dim=-1, keepdim=True)

        # ============ zero shot with features from CLIP and video head =================
        acc_split_tem, acc_split_top5_tem = multi_split_test(temporal_feat.cpu(), text_feat.cpu(), la.cpu())
        accuracy_split, accuracy_split_std = np.mean(acc_split_tem), np.std(acc_split_tem)
        accuracy_split_top5, accuracy_split_top5_std = np.mean(acc_split_top5_tem), np.std(acc_split_top5_tem)
        print('-----Half-classes Evaluation Temporal CLIP-----')
        print('Top1: mean {:.03f}%, std {:.03f}%'.format(accuracy_split, accuracy_split_std))
        print('Top5: mean {:.03f}%, std {:.03f}%'.format(accuracy_split_top5, accuracy_split_top5_std))
    return top1.avg


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output.cpu()

def compute_accuracy(vis_emb, text_emb, label):
    n_class = len(text_emb) # int: num_class
    n_samples = len(vis_emb) # int: num_video
    similarity=(100.0 * vis_emb @ text_emb.T) # [num_video, num_crop, num_class]

    # fuse then normalize
    similarity=similarity.mean(dim = 1, keepdim = False)  # b 101 [num_videos, num_classes]
    similarity=similarity.view(n_samples, n_class).softmax(dim = -1)
    
    # similarity: [num_videos, num_classes]
    prec=accuracy(similarity, label, topk = (1, 5))
    return prec[0], prec[1]
 
 
def multi_split_test(vis_embs, text_embs, true_label):
    full_acc1, full_acc5 = compute_accuracy(vis_embs, text_embs, true_label)
    print('-----Full-classes Evaluation------')
    print('Overall Top1 {:.03f}% Top5 {:.03f}%'.format(full_acc1.item(), full_acc5.item()))
 
    # Calculate accuracy per split
    # Only when the model has been trained on a different dataset
    true_label = true_label.numpy()
    accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
    for split in range(len(accuracy_split)):
        np.random.seed(split)
        sel_classes = np.random.permutation(len(text_embs))[:len(text_embs) // 2]  # [50, ]
        sel = [l in sel_classes for l in true_label]    # len = 10000 [<num_video]
        subclasses = np.unique(true_label[sel])         # [num_class//2 ]
        tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
        tl = torch.from_numpy(tl)
        acc, acc5 = compute_accuracy(vis_embs[sel], text_embs[subclasses], tl)
        accuracy_split[split] = acc
        accuracy_split_top5[split] = acc5
    
    return accuracy_split, accuracy_split_top5


if __name__ == '__main__':
    args = get_parser()
    main(args)

