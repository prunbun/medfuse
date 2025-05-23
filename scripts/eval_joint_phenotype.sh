CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python scripts/fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 5 --batch_size 16 --lr 5.652e-05 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr_cxr \
--fusion_type joint \
--save_dir /content/drive/MyDrive/medfuse_checkpoints/fusion_joint/eval