# system_emotion4
Body language system 

# Pre-requisite Libraries

* [Block1 GitHub repository (pre-processing)](https://github.com/trucomanx/OpenPifPafTools)
* [Block2 GitHub repository (face)](https://github.com/trucomanx/cnn_face_emotion)
* [Block3 GitHub repository (body)](https://github.com/trucomanx/cnn_emotion4)
* [Block4 GitHub repository (skeleton)](https://github.com/trucomanx/fcnn_emotion4)
* [Block5 GitHub repository (fusion)](https://github.com/trucomanx/fcnn_emotion4_fusion)

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


# Installation requirements summary for dataset BER2024

    ## Block 1: OpenPifPafTools
    git clone https://github.com/trucomanx/OpenPifPafTools.git
    cd OpenPifPafTools/src
    python3 setup.py sdist
    pip3 install dist/OpenPifPafTools-*.tar.gz
    
    ## Block 2: Face
    git clone https://github.com/trucomanx/cnn_face_emotion
    gdown 10PZUfBSJt3FXcNaA8UfvP6hGC46E0NoR
    unzip models.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    cd cnn_face_emotion/library
    python3 setup.py sdist
    pip3 install dist/FaceEmotion4Lib-*.tar.gz
    
    ## Block 3: Body
    git clone https://github.com/trucomanx/cnn_emotion4
    gdown 1TK6OPySP6NZGQyW2h8e_PHPRtDaz3s-X
    unzip models_2_v2.zip -d cnn_emotion4/library/BodyEmotion4Lib/models
    cd cnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/BodyEmotion4Lib-*.tar.gz
    
    ## Block 4: Skeleton
    git clone https://github.com/trucomanx/fcnn_emotion4
    gdown 10UtJHW0pETBKW6ptEZ1zzDVodwfwhZ8m
    unzip models_v2.zip -d fcnn_emotion4/library/SkeletonEmotion4Lib/models
    cd fcnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SkeletonEmotion4Lib-*.tar.gz
    
    ## Block 5: Fusion
    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    gdown 19I8TAOQhi2NMz-I81ih5Lz8zDXG-7y4O
    unzip models_fusion_v2.zip -d fcnn_emotion4_fusion/library/FusionEmotion4Lib/models
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz


# Installation requirements summary for dataset FULL2024


    ## Block 1: OpenPifPafTools
    git clone https://github.com/trucomanx/OpenPifPafTools.git
    cd OpenPifPafTools/src
    python3 setup.py sdist
    pip3 install dist/OpenPifPafTools-*.tar.gz


    ## Block 2: Face
    git clone https://github.com/trucomanx/cnn_face_emotion
    gdown 18ZTsD3FF0_1H3goacGPZwOgcLOKXhw0b
    unzip models_face_full.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    cd cnn_face_emotion/library
    python3 setup.py sdist
    pip3 install dist/FaceEmotion4Lib-*.tar.gz
    
    ## Block 3: Body
    git clone https://github.com/trucomanx/cnn_emotion4.git
    gdown 1_b2ppeKedwKNSDtILOTpreVnl7K3XFY5
    unzip models_body_full.zip -d cnn_emotion4/library/BodyEmotion4Lib/models
    cd cnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/BodyEmotion4Lib-*.tar.gz

    ## Block 4: Skeleton
    git clone https://github.com/trucomanx/fcnn_emotion4
    gdown 1EyKgM_SvNIW9OO8kK4IkPuACjdXoP-cv
    unzip models_skel_full.zip -d fcnn_emotion4/library/SkeletonEmotion4Lib/models
    cd fcnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SkeletonEmotion4Lib-*.tar.gz
    
    ## Block 5: Fusion
    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    gdown 1gk8BYQDDF_8t_IUC4tLjYXxdWjOFIWtE
    unzip models_fusion_full.zip -d fcnn_emotion4_fusion/library/FusionEmotion4Lib/models
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz

# Installation requirements summary for dataset FULL2024-DROP-FACE


    ## Block 1: OpenPifPafTools
    git clone https://github.com/trucomanx/OpenPifPafTools.git
    cd OpenPifPafTools/src
    python3 setup.py sdist
    pip3 install dist/OpenPifPafTools-*.tar.gz


    ## Block 2: Face
    git clone https://github.com/trucomanx/cnn_face_emotion
    gdown 18ZTsD3FF0_1H3goacGPZwOgcLOKXhw0b
    unzip models_face_full.zip -d cnn_face_emotion/library/FaceEmotion4Lib/models
    cd cnn_face_emotion/library
    python3 setup.py sdist
    pip3 install dist/FaceEmotion4Lib-*.tar.gz
    
    ## Block 3: Body
    git clone https://github.com/trucomanx/cnn_emotion4.git
    gdown 1_b2ppeKedwKNSDtILOTpreVnl7K3XFY5
    unzip models_body_full.zip -d cnn_emotion4/library/BodyEmotion4Lib/models
    cd cnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/BodyEmotion4Lib-*.tar.gz

    ## Block 4: Skeleton
    git clone https://github.com/trucomanx/fcnn_emotion4
    gdown 1EyKgM_SvNIW9OO8kK4IkPuACjdXoP-cv
    unzip models_skel_full.zip -d fcnn_emotion4/library/SkeletonEmotion4Lib/models
    cd fcnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SkeletonEmotion4Lib-*.tar.gz
    
    ## Block 5: Fusion
    git clone https://github.com/trucomanx/fcnn_emotion4_fusion
    gdown 1DFduCOACBDi7AM5rstCItWP3tRj7Jqc4
    unzip models_fusion_full.zip -d fcnn_emotion4_fusion/library/FusionEmotion4Lib/models
    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist
    pip3 install dist/FusionEmotion4Lib-*.tar.gz

# Installation summary

    git clone https://github.com/trucomanx/system_emotion4
    cd system_emotion4/library
    python3 setup.py sdist
    pip3 install dist/SystemEmotion4Lib-*.tar.gz



