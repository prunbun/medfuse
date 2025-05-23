CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python scripts/fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 \
--vision_num_classes 14 --num_classes 25 \
--data_pairs partial_ehr_cxr \
--fusion_type lstm \
--save_dir /content/drive/MyDrive/medfuse_checkpoints/fusion_partial/baseline/eval_optimal \
--load_state /content/drive/MyDrive/medfuse_checkpoints/their_best_checkpoints/optimal/best_checkpoint.pth.tar