# %%
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable cuda

# %%
input_default_json_conf_file='system_accuracy.json';

# %%
import sys

import hashlib
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import load_img

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
## Load json conf file
fd = open(os.path.join('./',input_default_json_conf_file));
DATA = json.load(fd);
fd.close()

# %%
"""
# Variable globales
"""

# %%
## Seed for the random variables
seed_number=0;

## Dataset 
dataset_base_dir     = DATA['dataset_base_dir'];
dataset_labels_file  = DATA['dataset_labels_file'];
dataset_name         = DATA['dataset_name'];

checkpoint      = DATA['checkpoint'];
model_type_body = DATA['model_type_body'];
model_type_face = DATA['model_type_face'];
model_type_skel = DATA['model_type_skel'];
model_type_fusion = DATA['model_type_fusion'];
enable_minus = DATA["enable_minus"];

# datafrem input dataset
label_colname='label';
filename_colname='filename';

## Output
output_base_dir = DATA['output_base_dir'];

sub_dir='subdir';
clean_break=200;

body_factor=DATA['body_factor'];
face_factor=DATA['face_factor'];
face_detector_method=DATA['face_detector_method'];

batch_size = 16

##############################################


# %%
"""
# If command line
"""

# %%
for n in range(len(sys.argv)):
    if   sys.argv[n]=='--dataset-dir':
        dataset_base_dir=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-file':
        dataset_labels_file=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-name':
        dataset_name=sys.argv[n+1];
    elif sys.argv[n]=='--model-type-check':
        checkpoint=sys.argv[n+1];
    elif sys.argv[n]=='--model-type-body':
        model_type_body=sys.argv[n+1];
    elif sys.argv[n]=='--model-type-face':
        model_type_face=sys.argv[n+1];
    elif sys.argv[n]=='--model-type-skel':
        model_type_skel=int(sys.argv[n+1]);
    elif sys.argv[n]=='--model-type-fusion':
        model_type_fusion=int(sys.argv[n+1]);
    elif sys.argv[n]=='--output-dir':
        output_base_dir=sys.argv[n+1];
    elif sys.argv[n]=='--sub-dir':
        sub_dir=sys.argv[n+1];
    elif sys.argv[n]=='--enable-minus':
        enable_minus=sys.argv[n+1].lower()=='true';
    elif sys.argv[n]=='--face-detector-method':
        face_detector_method=int(sys.argv[n+1]);
    elif sys.argv[n]=='--face-factor':
        face_factor=float(sys.argv[n+1]);
    elif sys.argv[n]=='--body-factor':
        body_factor=float(sys.argv[n+1]);
    elif sys.argv[n]=='--clean-break':
        clean_break=int(sys.argv[n+1]);

INFO=dict();
INFO["model_type_body"] = model_type_body;
INFO["model_type_face"] = model_type_face;
INFO["model_type_skel"] = model_type_skel;
INFO["model_type_fusion"] = model_type_fusion;
INFO["enable_minus"] = enable_minus;
INFO['body_factor'] = body_factor;
INFO['face_factor'] = face_factor;
INFO['face_detector_method'] = face_detector_method;

# Serializar o dicionário em uma string JSON com uma ordenação consistente
json_string = json.dumps(INFO, sort_keys=True)
# Gerar um hash MD5 a partir da string JSON
hash_object = hashlib.md5(json_string.encode())
sub_dir = hash_object.hexdigest()

print('   dataset_base_dir:',dataset_base_dir)
print('dataset_labels_file:',dataset_labels_file)
print('       dataset_name:',dataset_name)
print('    model_type_body:',model_type_body)
print('    model_type_face:',model_type_face)
print('    model_type_skel:',model_type_skel)
print('  model_type_fusion:',model_type_fusion)
print('    output_base_dir:',output_base_dir)
print('            sub_dir:',sub_dir)
print('       enable_minus:',enable_minus)
print('        clean_break:',clean_break)


# %%
"""
# Set seed of random variables

"""

# %%
np.random.seed(seed_number)
tf.keras.utils.set_random_seed(seed_number);

# %%
"""
# Loading data of dataset
"""

# %%
# Load filenames and labels
df_full = pd.read_csv(os.path.join(dataset_base_dir,dataset_labels_file));
print(df_full)

# %%
"""
# Creating output directory
"""

# %%

output_dir = os.path.join(output_base_dir,
                          dataset_name,
                          sub_dir,
                          'system_accuracy');

os.makedirs(output_dir,exist_ok = True);

# %%
# Salvando o dicionário como JSON
with open(os.path.join(output_dir,"system_info.json"), "w") as file:
    json.dump(INFO, file, indent=4)

# %%
sys.path.append('../library');

# %%
"""
# Dictionary
"""

# %%
Info=dict();    
Info["dataset_base_dir"] = dataset_base_dir;
Info["dataset_labels_file"] = dataset_labels_file;
Info["dataset_name"] = dataset_name;
Info["checkpoint"] = checkpoint;
Info["model_type_body"] = model_type_body;
Info["model_type_face"] = model_type_face;
Info["model_type_skel"] = model_type_skel;
Info["model_type_fusion"] = model_type_fusion;
Info["enable_minus"]=enable_minus;
Info['body_factor']=body_factor;
Info['face_factor']=face_factor;
Info['face_detector_method'] = face_detector_method;

import platform
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    gpu_info = gpu_details.get('device_name')
    Info["platform"]=gpu_info;
    print(f"Modelo da GPU: {gpu_info}")
else:
    cpu_info=platform.processor();
    Info["platform"]=cpu_info;
    print(f"Modelo da CPU: {cpu_info}")

print(json.dumps(Info, indent=4, ensure_ascii=False))

# %%
"""
# Create new model
"""

# %%
import SystemEmotion4Lib.Classifier as sec

cls=sec.Emotion4Classifier(checkpoint=checkpoint,
                           model_type_face=model_type_face,
                           model_type_body=model_type_body,
                           model_type_skel=model_type_skel,
                           model_type_skel_enable_minus=enable_minus,
                           model_type_fusion=model_type_fusion,
                           body_factor=body_factor,
                           face_factor=face_factor,
                           face_detector_method=face_detector_method);



# %%
"""
# Creating PIL data images
"""

# %%
from tqdm import tqdm
from IPython.display import display
from collections import Counter
import torch
import tensorflow as tf
import csv

Count=0;
N=0;
L=len(df_full);

output_filepath=os.path.join(output_dir,dataset_labels_file+".json");
output_filepath_csv=os.path.join(output_dir,dataset_labels_file+".output.csv");

initial_value=0;
if os.path.exists(output_filepath_csv):
    df = pd.read_csv(output_filepath_csv);
    initial_value = len(df);

#for index, row in tqdm(df_full.iterrows(), total=len(df_full), desc="Processing images"):
#for index, row in df_full.iterrows():
#pbar=tqdm(df_full.iterrows(), total=len(df_full), desc="Processing images");
pbar=tqdm(initial=initial_value, total=L, desc="Processing images");

if initial_value==0:
    with open(output_filepath_csv, mode='a', newline='') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow([filename_colname,label_colname,'predict','predict face','predict body','predict skeleton'])

for Count in range(initial_value,L,batch_size):
    # filepath
    filenames = df_full.iloc[Count:(Count+batch_size)][filename_colname].tolist();
    filepaths = [os.path.join(dataset_base_dir,fname) for fname in filenames];
    pil_imgs  = [load_img(filepath) for filepath in filepaths];

    # label_img
    label_imgs = df_full.iloc[Count:(Count+batch_size)][label_colname].tolist();
    label_imgs = [label.lower() for label in label_imgs];
    
    #pred, pred_face, pred_body, pred_skel = cls.from_img_all_pil(pil_img);
    res, res_face, res_body, res_skel, face_bbox_list, body_bbox_list= cls.predict_all_pil_list(pil_imgs);
    preds      = np.argmax(res     ,axis=1);
    preds_face = np.argmax(res_face,axis=1);
    preds_body = np.argmax(res_body,axis=1);
    preds_skel = np.argmax(res_skel,axis=1);

    reals = [cls.target_labels().index(label_img) for label_img in label_imgs];
    
    for pred,real,pred_face,pred_body,pred_skel,filename,label_img in zip(preds,reals,preds_face,preds_body,preds_skel,filenames,label_imgs):
        if pred == real:
            N += 1

        with open(output_filepath_csv, mode='a', newline='') as arquivo_csv:
            escritor = csv.writer(arquivo_csv)
            lista = [   filename,
                        label_img,
                        cls.target_labels()[pred],
                        cls.target_labels()[pred_face],
                        cls.target_labels()[pred_body]];
            if enable_minus:
                lista.append('unknown');
            else:
                lista.append(cls.target_labels()[pred_skel]);
            
            escritor.writerow(lista);

    if Count%clean_break==0:
        torch.cuda.empty_cache();
        tf.keras.backend.clear_session();
    
    pbar.update(len(filenames));
    pbar.set_description("fast acc:%.5f"%(N*1.0/(len(filenames)+Count-initial_value)))

#ssInfo["iterations_per_second"]=pbar.format_dict['rate'];

# %%
df = pd.read_csv(output_filepath_csv);

# Comparar as colunas 'colA' e 'colB'
iguais = df[label_colname] == df['predict'];
Info["length"]=L;
Info["match"]=int(iguais.sum());
Info["accuracy"] = float(iguais.mean());

iguais = df[label_colname] == df['predict face'];
Info["match_face"]=int(iguais.sum());
Info["accuracy_face"] = float(iguais.mean());

iguais = df[label_colname] == df['predict body'];
Info["match_body"]=int(iguais.sum());
Info["accuracy_body"] = float(iguais.mean());

iguais = df[label_colname] == df['predict skeleton'];
Info["match_skel"]=int(iguais.sum());
Info["accuracy_skel"] = float(iguais.mean());

# %%

print("ACC:",Info["accuracy"]);

print(json.dumps(Info, indent=4))

with open(output_filepath, "w") as arquivo_json:
    json.dump(Info, arquivo_json, indent=4, ensure_ascii=False)