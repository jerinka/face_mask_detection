import numpy as np
import argparse
import cv2
import pprint
import os


from Face_detect_opencv.face_det import FaceDet
from Person_MobilNet_SSD_opencv.person_det import PersonDet

if __name__=='__main__':
    face_obj = FaceDet()
    dets = face_obj.detect(image=None)
    pprint.pprint(dets)
    
    for det_count,box in dets.items():
        crop = box['crop']
        cv2.imshow('crop',crop)
        cv2.waitKey(100)
    
    
       
    person_obj = PersonDet()
    dets = person_obj.detect(frame=None)
    pprint.pprint(dets)
    
    for det_count,box in dets.items():
        crop = box['crop']
        cv2.imshow('crop',crop)
        cv2.waitKey(100)
