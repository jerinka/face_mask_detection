import numpy as np
import argparse
import cv2
import pprint
import os


from mask_det import MaskDet

class App:
    def __init__(self):
        self.path = os.path.abspath(os.path.dirname(__file__))
        # construct the argument parse 
        parser = argparse.ArgumentParser(
            description='Script to do mask detection')
        parser.add_argument("--image", default= self.abspath("img.jpeg"), help="path to image input")
        self.args = parser.parse_args()
        
        self.appobj=MaskDet()
    
    def 
    detsp = appobj.detect()
    pprint.pprint(detsp)
        
