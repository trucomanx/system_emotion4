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

import SystemEmotion4Lib.Classifier as sec
cls=sec.Emotion4Classifier( checkpoint='shufflenetv2k16',
                            model_type_face='efficientnet_b3',
                            model_type_body='efficientnet_b3',
                            model_type_skel=20,
                            model_type_skel_enable_minus=True,
                            model_type_fusion=7,
                            body_factor=1.0,
                            face_factor=0.85);

def my_batch_func(batch):
    
    img_pil_list=[];
    for item in batch:
        filepath=item[0];
        img_pil_list.append(Image.open(filepath));

    res=cls.from_img_pil_list(img_pil_list);
    
    words = ['negative', 'neutro', 'pain', 'positive']
    
    return [words[ID] for ID in res];


DIRECTORY = '/home/fernando/Downloads/output'

png_files = toolf.list_png_in_match_subdir(DIRECTORY,match_subdir='body',file_ext='.png');

res, file_list = toolf.verify_dataset_body_structure(png_files);

if res:
    print("Todos os",len(file_list), "arquivos existem.");
    toolf.save_dataset_list_in_csv_batch(DIRECTORY,file_list, 'output.csv', my_batch_func);
else:
    print("Arquivos com defeito:");
    with open('error.csv', 'w') as arquivo:
        arquivo.write('body,face,skeleton\n')
        for item in file_list:
            print(item);
            arquivo.write(item[0]+','+item[1]+','+item[2]+'\n')
    print("Achados",len(file_list),"arquivos com defeito. Ver error.csv");
    
    exit();





