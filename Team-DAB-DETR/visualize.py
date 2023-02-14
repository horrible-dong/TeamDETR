import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import util.misc as utils
from datasets import build_dataset
from main import get_args_parser, build_model_main
from util import box_ops
from util.visualizer import COCOVisualizer


@torch.no_grad()
def main(args):
    # qt +
    if args.q_splits is not None:
        args.q_splits = [int(i / 100 * args.num_queries) for i in args.q_splits]

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:' + str(n_parameters))

    dataset_val = build_dataset(image_set='val', args=args)
    id2name = {item['id']: item['name'] for item in dataset_val.coco.dataset['categories']}

    visualizer = COCOVisualizer()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.chmod(args.output_dir, 0o777)

    for i, (image, targets) in enumerate(tqdm(dataset_val)):
        # --------------------- visualize targets --------------------------
        box_label = [id2name[int(item)] for item in targets['labels']]
        gt_dict = {
            'boxes': targets['boxes'],
            'image_id': targets['image_id'],
            'size': targets['size'],
            'box_label': box_label,
        }
        visualizer.visualize(image, gt_dict, caption="gt", savedir=args.output_dir, show_in_console=False)
        # ------------------------------------------------------------------

        # --------------------- visualize predictions --------------------------
        targets = {k: v.to(device) for k, v in targets.items()}

        output = model(image[None].to(device))
        output = postprocessors['bbox'](output, torch.tensor([[1.0, 1.0]], device=device))[0]
        thershold = 0.3  # set a thershold

        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > thershold
        box_label = [id2name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            # qt +
            # "ref": model.refpoint_embed.weight.data,

            'boxes': boxes[select_mask],
            'image_id': targets['image_id'],
            'size': targets['size'],
            'box_label': box_label
        }
        visualizer.visualize(image, pred_dict, caption="pred", savedir=args.output_dir, show_in_console=False)
        # ------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args.output_dir = "../__visualize__"

    main(args)
