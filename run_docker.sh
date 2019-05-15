#!/bin/bash
# Arguments: GPU_ID EXPERIMENT

GPU_ID=$1
EXPERIMENT=$2

NV_GPU=$GPU_ID nvidia-docker run --rm -ti -v /data/research/wvr/Projects/AdversarialYOLO:/project -p 5815${GPU_ID}:6006 --name WVR_adversarial_yolo_${GPU_ID}_$EXPERIMENT -w /project nvcr.io/eavise/adversarial_yolo:v0 python train_patch.py $EXPERIMENT