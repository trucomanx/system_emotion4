#!/usr/bin/python3
import os

## Because the KERAS used is a old verion, KERAS 1
os.environ['TF_USE_LEGACY_KERAS'] = '1'

## Allows memory allocation on the GPU to be done asynchronously, 
## potentially improving performance and efficiency in certain scenarios.
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

## Specify which GPUs are available to a program.
## The list of GPUs is usually represented by indices starting at 0.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable cuda

import sys
sys.path.append('../library') 


import SystemEmotion4Lib.tools_funcs as toolf

def my_batch_func(batch):
    import random
    words = ['negative', 'neutro', 'pain', 'positive']
    
    return [random.choice(words) for ID in range(len(batch)) ]


DIRECTORY = '/home/fernando/Downloads/output/'

png_files = toolf.list_png_in_match_subdir(DIRECTORY,match_subdir='body',file_ext='.png');

res, file_list = toolf.verify_dataset_body_structure(png_files);

if res:
    print("Todos os",len(file_list), "arquivos existem.");
    toolf.save_dataset_list_in_csv_batch(file_list, 'output.csv', my_batch_func);
else:
    print("Arquivos faltando encontrados:");
    for files in file_list:
        print(files);
    exit();





