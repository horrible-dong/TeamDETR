coco_path=${1:-"../datasets/coco"}
output_dir=${2:-"./runs/__tmp__"}

torchrun --nproc_per_node=2 main.py \
	--output_dir $output_dir -c config/DINO/DINO_4scale.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
