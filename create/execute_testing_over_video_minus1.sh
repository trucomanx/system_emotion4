#!/bin/bash

MachinePath='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'

InTrD=$MachinePath"/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-VIDEOS/dataset-toy/drhouse_mini_cut.mp4"
DName="ber2024-source"

OutDir=$MachinePath"/OUTPUTS/DOCTORADO2/system_emotion4_1"

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert testing_over_video.ipynb testing_over_video.py


python3 testing_over_video.py   --model-type-check 'shufflenetv2k16' \
                                --model-type-face 'efficientnet_b3' \
                                --model-type-body 'efficientnet_b3' \
                                --model-type-skel 20 \
                                --model-type-fusion 7 \
                                --input-file $InTrD \
                                --dataset-name $DName \
                                --enable-minus true \
                                --face-detector-method 1 \
                                --body-factor 0.9 \
                                --face-factor 0.9 \
                                --output-dir $OutDir


rm -f testing_over_video.py
