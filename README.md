# system_emotion4
system_emotion4

# Using library
Since the code uses an old version of keras, it needs to be placed at the beginning of the main.py code.

    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    import SystemEmotion4Lib.Classifier as sec
    from PIL import Image
    
    cls=sec.Emotion4Classifier();

    img_pil = Image.new('RGB', (400,300), 'white');

    res=cls.from_img_pil(img_pil);

    print(res);


# Installation requirements summary

    ## OpenPifPafTools
    git clone https://github.com/trucomanx/OpenPifPafTools.git
    cd OpenPifPafTools/src
    python3 setup.py sdist
    pip3 install dist/OpenPifPafTools-*.tar.gz
    
    ## Face
    git clone https://github.com/trucomanx/cnn_face_emotion
    gdown 10PZUfBSJt3FXcNaA8UfvP6hGC46E0NoR
    unzip models.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    cd cnn_face_emotion/library
    python3 setup.py sdist
    pip3 install dist/FaceEmotion4Lib-*.tar.gz
    
    ## Body
    git clone https://github.com/trucomanx/cnn_emotion4
    gdown 1o_AFJOMwW4M-621HfkXI_nqU1nLZG8DB
    unzip models_2.zip -d cnn_emotion4/library/BodyEmotion4Lib/models
    cd cnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/BodyEmotion4Lib-*.tar.gz
    
    ## Skeleton
    git clone https://github.com/trucomanx/fcnn_emotion4
    cd fcnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SkeletonEmotion4Lib-*.tar.gz
    
    ## Fusion
    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz

# Installation summary

    git clone https://github.com/trucomanx/system_emotion4
    cd system_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SystemEmotion4Lib-*.tar.gz



