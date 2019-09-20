# USAGE
# python mask_rcnn_video.py --input videos/cats_and_dogs.mp4 --output output/cats_and_dogs_output.avi --mask-rcnn mask-rcnn-coco

# import the necessary packages
import numpy as np
import argparse
#import imutils
import time
import cv2
import os
import logging

# set the logging config
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
    help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-d", "--draw", default="y",
    help="draw image on screen (y/n)")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
    "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
    "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
    "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
logging.info("loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# initialize the video stream and pointer to output video file
# vs = cv2.VideoCapture(args["input"])
vs = cv2.VideoCapture("http://192.168.1.29:8081/")
#vs = cv2.VideoCapture(0)

#if (!vs.isOpened())  // check if succeeded to connect to the camera
#   CV_Assert("Cam open failed");


# vs.set(cv2.CAP_PROP_FRAME_WIDTH,1024);
# vs.set(cv2.CAP_PROP_FRAME_HEIGHT,768);
vs.set(cv2.CAP_PROP_FPS, 0.5)

writer = None

logging.info("confidence treshold: " + str(args['confidence']))

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    logging.info("trying to grab a frame")
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        logging.error("could not grab frame")
        break

    logging.info("a frame was grabbed")

    # construct a blob from the input frame and then perform a
    # forward pass of the Mask R-CNN, giving us (1) the bounding box
    # coordinates of the objects in the image along with (2) the
    # pixel-wise segmentation for each specific object
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final",
        "detection_masks"])
    end = time.time()

    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the
        # confidence (i.e., probability) associated with the
        # prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # detect cars only
        if classID not in [2]:
            continue

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the frame and then compute the width and the
            # height of the bounding box
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                interpolation=cv2.INTER_NEAREST)
            mask = (mask > args["threshold"])

            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]

            # grab the color used to visualize this particular class,
            # then create a transparent overlay by blending the color
            # with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                color, 2)

            # draw the predicted label and associated probability of
            # the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

            logging.info("Car confidence: " + str(confidence))

    # show the image
    if args['draw'] == 'y':
        cv2.imshow("mask_rcnn", frame)

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release the file pointers
logging.info("cleaning up...")

# When everything done, release the capture
vs.release()
cv2.destroyAllWindows()
