#!/bin/bash

MachinePath='/media/fernando/Expansion'

InTrD=$MachinePath"/DATASET/TESE-PATIENT-RECOGNITION/PATIENT-VIDEOS/dataset-toy/drhouse_mini_cut.mp4"
DName="full2024-minus-drop-plus"

OutDir=$MachinePath"/OUTPUTS/DOCTORADO2/system_emotion4_full_1"

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert testing_over_video.ipynb testing_over_video.py


python3 testing_over_video.py   --model-type-check 'shufflenetv2k16' \
                                --model-type-face 'efficientnet_b3' \
                                --model-type-body 'efficientnet_b3' \
                                --model-type-skel 81 \
                                --model-type-fusion 39 \
                                --input-file $InTrD \
                                --dataset-name $DName \
                                --enable-minus true \
                                --face-detector-method 0 \
                                --body-factor 1.0 \
                                --face-factor 1.0 \
                                --batch-size-func 16 \
                                --output-dir $OutDir

rm -f testing_over_video.py
