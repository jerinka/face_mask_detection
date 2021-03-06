#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
import pprint
import os

class PersonDet:
    def __init__(self):
        
        # construct the argument parse 
        parser = argparse.ArgumentParser(
            description='Script to run MobileNet-SSD object detection network')
        parser.add_argument("--image", default= self.abspath("img.jpeg"), help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", default=self.abspath("MobileNetSSD_deploy.prototxt"),
                                          help='Path to text network file: '
                                               'MobileNetSSD_deploy.prototxt for Caffe model'
                                               )
        parser.add_argument("--weights", default=self.abspath("MobileNetSSD_deploy.caffemodel"),
                                         help='Path to weights: '
                                              'MobileNetSSD_deploy.caffemodel for Caffe model'
                                              )
        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
        self.args = parser.parse_args()
        
        # Labels of Network.
        self.classNames = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

        #Load the Caffe model 
        self.net = cv2.dnn.readNetFromCaffe(self.args.prototxt, self.args.weights)

    def abspath(self,filename):
        self.path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(self.path, filename)
        
    def detect(self, frame=None, show=True,wanted_class = ['person']):    
        # Load image fro
        if frame is None:
            frame = cv2.imread(self.args.image)
        
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
        heightFactor = frame.shape[0]/300.0
        widthFactor = frame.shape[1]/300.0 
        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        self.net.setInput(blob)
        #Prediction of network
        detections = self.net.forward()
                    

        frame_copy = frame.copy()
        frame_copy2 = frame.copy()
        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]
        
        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        dets = {}
        det_count=0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > self.args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label
                
                if self.classNames[class_id] in wanted_class:

                    # Object location 
                    xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop   = int(detections[0, 0, i, 5] * cols)
                    yRightTop   = int(detections[0, 0, i, 6] * rows)

                    xLeftBottom_ = int(widthFactor * xLeftBottom) 
                    yLeftBottom_ = int(heightFactor* yLeftBottom)
                    xRightTop_   = int(widthFactor * xRightTop)
                    yRightTop_   = int(heightFactor * yRightTop)
                    
                    crop = frame[yLeftBottom_:yRightTop_,xLeftBottom_:xRightTop_]
                    
                    
                    dets[det_count] = {'class_id':self.classNames[class_id], 'xLeftBottom_':xLeftBottom_,'yLeftBottom_':yLeftBottom_,'xRightTop_':xRightTop_,'yRightTop_':yRightTop_,'confidence':confidence, 'crop':crop}
                    
                    #import pdb;pdb.set_trace()

                    cv2.rectangle(frame_copy, (dets[det_count]['xLeftBottom_'], yLeftBottom_), (xRightTop_, yRightTop_),(0, 255, 0),-1)
                    det_count+=1
                    
                             
        opacity = 0.3
        cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)

        for det_count,box in dets.items():
            

                xLeftBottom_ = box['xLeftBottom_']
                yLeftBottom_ = box['yLeftBottom_']
                xRightTop_   = box['xRightTop_']
                yRightTop_   = box['yRightTop_']
                cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),(0, 0, 0),2)
                # Draw label and confidence of prediction in frame resized
                if class_id in self.classNames:
                    label = box['class_id'] + ": " + str(box['confidence'])
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)

                    yLeftBottom_ = max(yLeftBottom_, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_ - labelSize[1]),
                                         (xLeftBottom_ + labelSize[0], yLeftBottom_ + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom_, yLeftBottom_),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
        if show == True:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", frame)
            cv2.waitKey(500)
        return dets

if __name__ == '__main__':
    person_obj = PersonDet()
    dets = person_obj.detect(frame=None)
    pprint.pprint(dets)
    
    for det_count,box in dets.items():
        crop = box['crop']
        cv2.imshow('crop',crop)
        cv2.waitKey(0)
    
