#!/bin/bash

MachinePath='/media/fernando/Expansion/'

InTrD=$MachinePath"/DATASET/TESE/BER/BER2024-SOURCE"
DName="ber2024-source"

OutDir=$MachinePath"/OUTPUTS/DOCTORADO2/system_emotion4_1"
SubDir="subdir1"

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert system_accuracy.ipynb system_accuracy.py

python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 20 \
                            --model-type-fusion 11 \
                            --dataset-dir $InTrD \
                            --dataset-file "train.csv" \
                            --dataset-name $DName \
                            --sub-dir $SubDir \
                            --output-dir $OutDir

python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 20 \
                            --model-type-fusion 11 \
                            --dataset-dir $InTrD \
                            --dataset-file "test.csv" \
                            --dataset-name $DName \
                            --sub-dir $SubDir \
                            --output-dir $OutDir

rm -f system_accuracy.py
