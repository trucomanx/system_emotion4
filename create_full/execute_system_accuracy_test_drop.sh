#!/bin/bash

MachinePath='/media/fernando/Expansion'

OutDir=$MachinePath"/OUTPUTS/DOCTORADO2/SYSTEM/system_emotion4_full_minus_drop_plus"

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert system_accuracy_batch.ipynb system_accuracy.py


python3 system_accuracy.py  --model-type-check 'shufflenetv2k16' \
                            --model-type-face 'efficientnet_b3' \
                            --model-type-body 'efficientnet_b3' \
                            --model-type-skel 81 \
                            --model-type-fusion 39 \
                            --dataset-dir $MachinePath"/DATASET/TESE-DROP-FACE10" \
                            --dataset-file "test_body.csv" \
                            --dataset-name "full2024-body-drop-face10" \
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
                            --model-type-fusion 39 \
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
