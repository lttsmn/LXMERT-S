# The name of this experiment.
name=$2

# Save logs and models under snap/nlvr2; make backup.
output=snap/nlvr2/$name
mkdir -p $output/src_step2
cp -r src_step2/* $output/src_step2/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src_step2 \
    python src/tasks/nlvr2.py \
    --tiny --llayers 0 --xlayers 5 --rlayers 0 \
    --tqdm --output $output ${@:3}
