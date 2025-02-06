#!/bin/bash

MachinePath='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando'

OutDir=$MachinePath"/OUTPUTS/DOCTORADO2/SYSTEM/system_emotion4_full"

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert system_accuracy_batch.ipynb system_accuracy.py


python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 81 \
                            --model-type-fusion 11 \
                            --dataset-dir $MachinePath"/DATASET/TESE-DROP-FACE" \
                            --dataset-file "test_body.csv" \
                            --dataset-name "full2024-body-drop-face" \
                            --enable-minus true \
                            --face-detector-method 0 \
                            --body-factor 1.0 \
                            --face-factor 1.0 \
                            --clean-break 100 \
                            --output-dir $OutDir

python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 81 \
                            --model-type-fusion 11 \
                            --dataset-dir $MachinePath"/DATASET/TESE" \
                            --dataset-file "test_body.csv" \
                            --dataset-name "full2024-body" \
                            --enable-minus true \
                            --face-detector-method 0 \
                            --body-factor 1.0 \
                            --face-factor 1.0 \
                            --clean-break 100 \
                            --output-dir $OutDir


rm -f system_accuracy.py
