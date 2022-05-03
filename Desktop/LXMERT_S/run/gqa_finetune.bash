# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train,valid --valid testdev \
    --llayers 0 --xlayers 5 --rlayers 0 \
    --loadLXMERT data/model  \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 5 \
    --tqdm --output $output ${@:3}
