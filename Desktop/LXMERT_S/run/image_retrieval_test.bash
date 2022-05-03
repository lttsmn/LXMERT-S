# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src_step2
cp -r src_step2/* $output/src_step2/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src_step2 \
    python src_step2/tasks/image_retrieval.py \
    --train "" --valid "" \
    --llayers 0 --xlayers 5 --rlayers 0 \
    --batchSize 500 --optim bert --lr 5e-5 --epochs 8 \
    --tqdm --output $output ${@:3}
