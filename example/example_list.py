#!/usr/bin/python3

import os

## Because the KERAS used is a old verion, KERAS 1
os.environ['TF_USE_LEGACY_KERAS'] = '1'

## Allows memory allocation on the GPU to be done asynchronously, 
## potentially improving performance and efficiency in certain scenarios.
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

## Specify which GPUs are available to a program.
## The list of GPUs is usually represented by indices starting at 0.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable cuda

import sys
sys.path.append('../library') 


import SystemEmotion4Lib.Classifier as sec
import numpy as np
from PIL import Image
import time

cls=sec.Emotion4Classifier( checkpoint='shufflenetv2k16',
                            model_type_face='efficientnet_b3',
                            model_type_body='efficientnet_b3',
                            model_type_skel=20,
                            model_type_skel_enable_minus=True,
                            model_type_fusion=7);


labels   = ['negative','neutral','pain','positive'];
filepaths = [   'KlingAi/negative.png',
                'KlingAi/neutral.png',
                'KlingAi/pain.png',
                'KlingAi/positive.png',
                'KlingAi/pain_negative.png'];

img_pil_list=[];
for filepath in filepaths:
    img_pil_list.append(Image.open(filepath));


print("")
tic = time.time()
for n in range(len(filepaths)):
    filepath= filepaths[n];
    res=cls.predict_pil(img_pil_list[n]);
    print(filepath,res);
toc = time.time();
print(f"Tempo de execução: {toc - tic:.4f} segundos");




print("")
tic = time.time()
res=cls.predict_pil_list(img_pil_list);
for n in range(len(filepaths)):
    filepath= filepaths[n];
    print(filepath,res[n]);
toc = time.time();
print(f"Tempo de execução: {toc - tic:.4f} segundos");


