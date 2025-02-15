cd open_clip/src
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --train-data '/workspace/cc3m/{00000..00331}.tar' \
    --train-num-samples 3318333 \
    --model ViT-B-32 \
    --dataset-type webdataset \
    --batch-size 480 \
    --precision amp \
    --workers 4