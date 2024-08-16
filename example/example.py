#!/usr/bin/python3


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import SystemEmotion4Lib.Classifier as sec
from PIL import Image

cls=sec.Emotion4Classifier();

labels   = ['negative','neutral','pain','positive'];
filepath = 'dataset/positive.png';

img_pil = Image.open(filepath)

res=cls.from_img_pil(img_pil);

print(filepath,labels[res]);


