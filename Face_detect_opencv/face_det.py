# importing necessary packages
import numpy as np
import argparse
import cv2
import pprint

class FaceDet:

    def __init__(self):

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()

        ap.add_argument("-i", "--image", default='test_image01.jpg', help="patho to input image")
        ap.add_argument("-p", "--prototxt", default='deploy.prototxt.txt', help="path to Caffee 'deploy' prototxt file")
        ap.add_argument("-m", "--model", default='res10_300x300_ssd_iter_140000.caffemodel', help="path to Caffe pre-trained model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

        self.args = vars(ap.parse_args())

        # load model from disk
        print("[INFO] loading from model...")
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])

    def detect(self,image=None,show=True):
    
        # load the input image and construct an input blob for the image and resize image to
        # fixed 300x300 pixels and then normalize it
        if image is None:
            image = cv2.imread(self.args["image"])
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()
        dets={}
        det_count=0
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                crop = image[startY:endY, startX:endX]
                
                #import pdb;pdb.set_trace()
                dets[det_count] = {'class_id':'face', 'startX':startX,'startY':startY,'endX':endX,'endY':endY,'confidence':confidence, 'crop':crop}

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                det_count+=1
        if show==True:
            # show the output image
            cv2.imshow("Output", image)
            cv2.waitKey(0)
        return dets

if __name__=='__main__':
    face_obj = FaceDet()
    dets = face_obj.detect(image=None)
    pprint.pprint(dets)
    
    for det_count,box in dets.items():
        crop = box['crop']
        cv2.imshow('crop',crop)
        cv2.waitKey(0)
    
    
