import numpy as np
import argparse
import cv2
import pprint
import os


from Face_detect_opencv.face_det import FaceDet
from Person_MobilNet_SSD_opencv.person_det import PersonDet
from MaskClassifier.classifier import MaskClassifier

class MaskDet:
    def __init__(self):
        self.path = os.path.abspath(os.path.dirname(__file__))
        # construct the argument parse 
        parser = argparse.ArgumentParser(
            description='Script to do mask detection')
        parser.add_argument("--image", default= self.abspath("images/Trump.jpg"), help="path to image input")
        self.args = parser.parse_args()
        
        # Person detection
        self.person_obj = PersonDet()
        #Face detection
        self.face_obj = FaceDet()
        
        #face-mask classifier
        self.mask_classifier = MaskClassifier() 

    def abspath(self,filename):
        return os.path.join(self.path, filename)
        
    def detect(self, frame=None):
        if frame is None:
            frame = cv2.imread(self.args.image)
        
        #import pdb;pdb.set_trace()
           
        dets1 = self.person_obj.detect(frame=frame.copy())
        pprint.pprint(dets1)
        
        detsp={}
        det_countp=0
        for det_countp,boxp in dets1.items():
            crop = boxp['crop']
            H,W,_=crop.shape
            H=min(H,W)
            W=H
            
            crop = crop[0:H,0:W]
            #cv2.imshow('crop1',crop)
            #cv2.waitKey(100)
            
            x1p   = boxp['xLeftBottom_']
            y1p   = boxp['yLeftBottom_']
            x2p   = boxp['xRightTop_']
            y2p   = boxp['yRightTop_']
            
            dets2 = self.face_obj.detect(image=crop)
            pprint.pprint(dets2)
            
            detsf={}
            det_countf=0
            maskstatus=True
            for det_countf,boxf in dets2.items():
                x1 = boxp['xLeftBottom_']+boxf['startX']
                y1 = boxp['yLeftBottom_']+boxf['startY']
                x2 = boxp['xLeftBottom_']+boxf['endX']
                y2 = boxp['yLeftBottom_']+boxf['endY']
  
                crop = frame[y1:y2,x1:x2]
                
                classname,confidence = self.mask_classifier.predict_opencv_image(crop)
                
                print('classname: ',classname)
                if classname=='nomask':
                    maskstatus = False
                #import pdb;pdb.set_trace()
                
                detsf[det_countf] = {'class_id':'face', 'x1':x1,'y1':y1,'x2':x2,'y2':y2, 'crop':crop,'status':classname,'confidence':confidence}
                det_countf+=1
                
                
                if classname=='yesmask': 
                    cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0),2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 0, 255),2)
                
                cv2.imshow('frame4',frame)
                cv2.waitKey(100)
                
            if maskstatus==True: 
                cv2.rectangle(frame, (x1p, y1p), (x2p, y2p),(0, 255, 0),2)
            else:
                cv2.rectangle(frame, (x1p, y1p), (x2p, y2p),(0, 0, 255),2)
            
            detsp[det_countp]={'class':'person','x1p':x1p,'y1p':y1p,'x2p':x2p,'y2p':y2p,'faces':detsf,'maskstatus':maskstatus}
            
        cv2.imshow('frame4',frame)
        cv2.waitKey(0)
        return detsp
        
if __name__ == '__main__':
    appobj=MaskDet()
    detsp = appobj.detect()
    pprint.pprint(detsp)
    
    
