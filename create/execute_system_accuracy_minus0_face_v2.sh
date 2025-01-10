#!/bin/bash

MachinePath='/media/fernando/Expansion'

InTrD=$MachinePath"/DATASET/TESE/BER/BER2024-SOURCE"
DName="ber2024-source"

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert system_accuracy_batch.ipynb system_accuracy.py

OutDir=$MachinePath"/OUTPUTS/DOCTORADO2/system_emotion4_v2_factor_face"

for j in 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25
do

python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 53 \
                            --model-type-fusion 9 \
                            --dataset-dir $InTrD \
                            --dataset-file "train_refface.csv" \
                            --dataset-name $DName \
                            --enable-minus true \
                            --face-detector-method 0 \
                            --body-factor 1.0 \
                            --face-factor $j \
                            --clean-break 100 \
                            --output-dir $OutDir

python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 53 \
                            --model-type-fusion 9 \
                            --dataset-dir $InTrD \
                            --dataset-file "test_refface.csv" \
                            --dataset-name $DName \
                            --enable-minus true \
                            --face-detector-method 0 \
                            --body-factor 1.0 \
                            --face-factor $j \
                            --clean-break 100 \
                            --output-dir $OutDir

done


rm -f system_accuracy.py
