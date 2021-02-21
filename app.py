import numpy as np
import argparse
import cv2
import pprint
import os


from mask_det import MaskDet
import math
class App:
    def __init__(self):
        self.path = os.path.abspath(os.path.dirname(__file__))
        # construct the argument parse 

        self.maskobj=MaskDet()
        
    def abspath(self,filename):
        return os.path.join(self.path, filename)
        
    def check_socialdistancing(self,frame,dist_thr):
        detsp = self.maskobj.detect(frame.copy())
        pprint.pprint(detsp)
        violations=0
        for person_id1, boxinfo1 in detsp.items():
            x1 =  (boxinfo1['x1p']+boxinfo1['x2p'])/2
            y1 =  (boxinfo1['y1p']+boxinfo1['y2p'])/2
            for face_id1, faceinfo1 in boxinfo1['faces'].items():
                fx1 = faceinfo1['x1']
                fx2 = faceinfo1['x2']
                fy1 = faceinfo1['y1']
                fy2 = faceinfo1['y2']
                classname = faceinfo1['status']
                if classname == 'nomask':
                    cv2.rectangle(frame, (fx1,fy1), (fx2,fy2),(0, 0, 255),2)
                else:
                    cv2.rectangle(frame, (fx1,fy1), (fx2,fy2),(0, 255, 0),2)
                
            alert=False
            warn=False
            for person_id2, boxinfo2 in detsp.items():
                if person_id1!=person_id2:
                    x2 =  (boxinfo2['x1p']+boxinfo2['x2p'])/2
                    y2 =  (boxinfo2['y1p']+boxinfo2['y2p'])/2
                    maskstatus1 = boxinfo1['maskstatus']
                    maskstatus2 = boxinfo2['maskstatus']
                    dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
                    print('dist',dist)
                    
                    if dist<dist_thr and (maskstatus1==False or maskstatus2==False):
                        warn=True
                        cv2.rectangle(frame, (boxinfo1['x1p'], boxinfo1['y1p']), (boxinfo1['x2p'], boxinfo1['y2p']),(0, 0, 255),2)
                        #import pdb;pdb.set_trace()
                        
                        if dist<dist_thr and maskstatus1==False:
                            alert=True
                            violations+=1
                            label = 'Alert'
                            cv2.putText(frame, label, (boxinfo1['x1p'], boxinfo1['y1p']),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255),2)
                                                   
            if alert==False and warn==False:
                cv2.rectangle(frame, (boxinfo1['x1p'], boxinfo1['y1p']), (boxinfo1['x2p'], boxinfo1['y2p']),(0, 255, 0),2)
            
            cv2.imshow('result',frame)
        cv2.waitKey(0)   
        return violations
        
        
if __name__=='__main__':
    app = App()
    parser = argparse.ArgumentParser(
    description='Script to do mask detection')
    parser.add_argument("--image", default= "images/Trump.jpg", help="path to image input")  #Change this path to other images
    args = parser.parse_args()
    frame = cv2.imread(args.image)
    
    violations = app.check_socialdistancing(frame,dist_thr=198)
    print('Violations',violations)
    
    
        
        
