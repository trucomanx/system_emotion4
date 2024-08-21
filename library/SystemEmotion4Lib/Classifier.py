#!/usr/bin/python

import sys
import numpy as np

import OpenPifPafTools.RoiDetector    as opp
import FaceEmotion4Lib.Classifier     as fec
import BodyEmotion4Lib.Classifier     as bec
import SkeletonEmotion4Lib.Classifier as sec
import FusionEmotion4Lib.Classifier   as fsc

import tensorflow as tf
import numpy as np

class Emotion4Classifier:
    """Class to classify 4 body languages.
    
    The class Emotion4Classifier classify daa in 4 body languages.
    
    Args:
        :param file_of_weight: Archivo donde se encuentran los pesos.
        
    Atributos:
        modelo: Model returned by tensorflow.
    """
    def __init__(   self,
                    checkpoint='shufflenetv2k16',
                    model_type_face='efficientnet_b3',
                    model_type_body='efficientnet_b3',
                    model_type_skel=20,
                    model_type_skel_enable_minus=False,
                    model_type_fusion=11):
        """Inicializer of class Emotion4Classifier.
        
        Args:
            param file_of_weight: Archivo donde se encuentran los pesos.
        """
        


        self.cls_face=fec.FaceEmotion4Classifier(model_type=model_type_face);

        self.cls_body=bec.Emotion4Classifier(model_type=model_type_body);

        self.cls_skel=sec.Emotion4Classifier(ncod=model_type_skel);
        
        #mejor esa orden para que cargue al final
        self.det=opp.Detector(checkpoint='shufflenetv2k16');

        if model_type_skel_enable_minus==True:
            self.cls_fusion=fsc.Emotion4Classifier(ncod=model_type_fusion, skel_size=model_type_skel);
        else:
            self.cls_fusion=fsc.Emotion4Classifier(ncod=model_type_fusion, skel_size=None);
        
        self.enable_minus=model_type_skel_enable_minus;
        
    def get_input_fusion_from_pil(self,pil_img):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img: PIL image 
        
        Returns:
            numpy.array: A numpy array of 4 elements.
        """
        skel_vec, body_roi, face_roi=self.det.process_image(pil_img);
        if skel_vec is None:
            print('Error because skel_vec is None');
            sys.exit();
        
        res_body=np.array([0.0,0.0,0.0,0.0]);
        res_face=np.array([0.0,0.0,0.0,0.0]);
                
        if body_roi is not None:
            res_body = self.cls_body.predict_pil(body_roi);
        
        if face_roi is not None:
            res_face = self.cls_face.predict_pil(face_roi);
        
        if self.enable_minus==True:
            res_skel = self.cls_skel.predict_minus_vec(skel_vec);
        else:
            res_skel = self.cls_skel.predict_vec(skel_vec);
        
        return res_face, res_body, res_skel;

    def predict_pil(self,pil_img):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img: PIL image 
        
        Returns:
            numpy.array: A numpy array of 4 elements.
        """
        
        res_face, res_body, res_skel = self.get_input_fusion_from_pil(pil_img);

        fusion_vec = np.concatenate((res_face, res_body, res_skel));
        
        res=self.cls_fusion.predict_vec(fusion_vec);
        
        return res;

    def from_img_pil(self,pil_img):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img: PIL image 
        
        Returns:
            int: The class of image.
        """
        return np.argmax(self.predict_pil(pil_img));

    def target_labels(self):
        """Returns the categories of classifier.
        
        Returns:
            list: The labels of categories resturned by the methods from_skeleton_npvector().
        """
        return ['negative','neutro','pain','positive'];

    def dummy_test(self):
        # Definindo o modelo
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(4,))
        ]);

        # Compilando o modelo (opcional neste caso, pois não vamos treinar)
        model.compile(optimizer='adam', loss='mse');

        # Criando um vetor de entrada com 4 elementos
        input_data = np.array([[0.5, 0.3, 0.7, 0.2]]);

        # Fazendo uma previsão com o modelo
        prediction = model.predict(input_data);
        
        print('')
        print(prediction)
        print('')
