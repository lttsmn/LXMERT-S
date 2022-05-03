# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/sent_retrieval_zero_shot/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/sent_retrieval_test.py \
    --test test \
    --zero_shot \
    --llayers 0 --xlayers 5 --rlayers 0 \
    --loadLXMERTQA data/model \
    --batchSize 500 --optim bert --lr 5e-5 --epochs 8 \
    --tqdm --output $output ${@:3}
