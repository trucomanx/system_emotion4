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
                    model_type_fusion=11,
                    body_factor=1.0, 
                    face_factor=1.0,
                    verbose=False):
        """Inicializer of class Emotion4Classifier.
        
        Args:
            param file_of_weight: Archivo donde se encuentran los pesos.
        """
        


        self.cls_face=fec.FaceEmotion4Classifier(model_type=model_type_face);

        self.cls_body=bec.Emotion4Classifier(model_type=model_type_body);

        self.cls_skel=sec.Emotion4Classifier(ncod=model_type_skel);
        
        #mejor esa orden para que cargue al final
        self.det=opp.Detector(checkpoint='shufflenetv2k16', body_factor=body_factor, face_factor=face_factor,face_method=1);

        if model_type_skel_enable_minus==True:
            self.cls_fusion=fsc.Emotion4Classifier(ncod=model_type_fusion, skel_size=model_type_skel);
        else:
            self.cls_fusion=fsc.Emotion4Classifier(ncod=model_type_fusion, skel_size=None);
        
        self.enable_minus=model_type_skel_enable_minus;
        
        self.verbose=verbose;
        
    def get_input_fusion_from_pil(self,pil_img):
        """ Retorna a resposta de todos os sistemas bas e os bounding box:
            res_face, res_body, res_skel, face_bbox, body_bbox.
            Se nao se achou pessoas entao todos seus elementos sao None.
            Se se achou uma pessoa, entaoa  saida  e' trabalhavel.
            Se face nao 'e achado se retorna zeros e bouding box None.
            Se body nao 'e achado se retorna zeros e bouding box None.
        
        Args:
            pil_img: PIL image 
        
        Returns:
            numpy.array: A numpy array of 4 elements.
        """
        skel_vec, body_roi, face_roi, body_bbox, face_bbox=self.det.process_image_full(pil_img);
        if skel_vec is None:
            if self.verbose:
                print('No person was found, will be returned All None.');
            return None, None, None, None, None;
        
        res_body=np.array([0.0,0.0,0.0,0.0]);
        res_face=np.array([0.0,0.0,0.0,0.0]);
                
        if body_roi is not None:
            res_body = self.cls_body.predict_pil(body_roi);
        else:
            if self.verbose:
                print('Body roi is None, body classifier return zeros.')
        
        if face_roi is not None:
            res_face = self.cls_face.predict_pil(face_roi);
        else:
            if self.verbose:
                print('Face roi is None, face classifier return zeros.')
        
        if self.enable_minus==True:
            res_skel = self.cls_skel.predict_minus_vec(skel_vec);
        else:
            res_skel = self.cls_skel.predict_vec(skel_vec);
        
        return res_face, res_body, res_skel, face_bbox, body_bbox;

    def predict_pil(self,pil_img):
        """Classify a body language data from a Pil image.
           Return None if any person was not found.
        
        Args:
            pil_img: PIL image 
        
        Returns:
            numpy.array: A numpy array of 4 elements.
        """
        res, _, _, _, _, _ = self.predict_all_pil(pil_img);
        
        return res;

    def predict_all_pil(self,pil_img):
        """Classify a body language data from a Pil image.
           Return None if any person was not found.
        
        Args:
            pil_img: PIL image 
        
        Returns:
            numpy.array, numpy.array, numpy.array, numpy.array, couple, couple : If any person was not found return all None.
            In other cases, 4 numpy array of 4 elements and 2 couples with bounding boxs.
            If body roi was None the bounding box will be None.
            If face roi was None the bounding box will be None.
        """
        
        res_face, res_body, res_skel, face_bbox, body_bbox = self.get_input_fusion_from_pil(pil_img);
        if res_skel is None:
            if self.verbose:
                print('No person was found, will be returned All None.');
            return None, None, None, None, None, None ;
        
        fusion_vec = np.concatenate((res_face, res_body, res_skel));
        
        res=self.cls_fusion.predict_vec(fusion_vec);
        
        return res, res_face, res_body, res_skel, face_bbox, body_bbox;

    def get_input_fusion_from_pil_list(self,pil_img_list):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img_list: List of PIL image 
        
        Returns:
            numpy.array: A numpy matrix of 4 columns.
        """
        skel_vec_list, body_roi_list, face_roi_list, body_bbox_list, face_bbox_list = self.det.process_image_full_list(pil_img_list);
        
        # Verifica errores
        for skel_vec in skel_vec_list:
            if skel_vec is None:
                print('Error because skel_vec is None');
                sys.exit();
        
        ##      
        res_body_mat = self.cls_body.predict_pil_list(body_roi_list);
        
        res_face_mat = self.cls_face.predict_pil_list(face_roi_list);
        
        if self.enable_minus==True:
            res_skel_mat = self.cls_skel.predict_minus_vec_list(skel_vec_list);
        else:
            res_skel_mat = self.cls_skel.predict_vec_list(skel_vec_list);
        
        return res_face_mat, res_body_mat, res_skel_mat, face_bbox_list, body_bbox_list;

    def predict_pil_list(self,pil_img_list):
        res, _, _, _, _, _ = self.predict_all_pil_list(pil_img_list);
        return res;
        
    def predict_all_pil_list(self,pil_img_list):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img: List of PIL image 
        
        Returns:
            numpy.array: A numpy matrix of 4 columns.
        """
        
        res_face_mat, res_body_mat, res_skel_mat, face_bbox_list, body_bbox_list = self.get_input_fusion_from_pil_list(pil_img_list);
        
        fusion_mat = np.concatenate((res_face_mat, res_body_mat, res_skel_mat),axis=1);
        
        res=self.cls_fusion.predict_mat(fusion_mat);
        
        return res, res_face_mat, res_body_mat, res_skel_mat, face_bbox_list, body_bbox_list;


    def from_img_pil(self,pil_img):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img: PIL image 
        
        Returns:
            int: The class of image.
        """
        return np.argmax(self.predict_pil(pil_img));
        

    def from_img_pil_list(self,pil_img_list):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img_list: List of PIL image 
        
        Returns:
            numpy.array: The class of image.
        """
        return np.argmax(self.predict_pil_list(pil_img_list),axis=1);

    def from_img_all_pil(self,pil_img):
        """Classify a body language data from a numpy vector object with N elements 
        
        Args:
            pil_img: PIL image 
        
        Returns:
            int: The class of image.
        """
        res, res_face, res_body, res_skel = self.predict_all_pil(pil_img);
        return np.argmax(res), np.argmax(res_face), np.argmax(res_body), np.argmax(res_skel);

    def target_labels(self):
        """Returns the categories of classifier.
        
        Returns:
            list: The labels of categories resturned by the methods from_skeleton_npvector().
        """
        return ['negative','neutro','pain','positive'];

