CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python scripts/fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 5 --batch_size 16 --lr 0.00007 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr_cxr \
--fusion_type lstm \
--save_dir /content/drive/MyDrive/medfuse_checkpoints/fusion_partial/reversed \
# --load_state_cxr /content/drive/MyDrive/medfuse_checkpoints/encoders/cxr_cnn/best_checkpoint.pth.tar \
# --load_state_ehr /content/drive/MyDrive/medfuse_checkpoints/encoders/ehr_lstm/best_checkpoint.pth.tar