
import numpy as np
import time
import cv2
import os


def main():
	print('hello!')

	#######init files
	basedir = os.getcwd()
	modeldir = os.path.join(basedir, 'model')
	datadir = os.path.join(basedir, 'bdd100k/images/10k')
	trainingdir = os.path.join(datadir, 'train')
	testdir = os.path.join(datadir, 'test')

	labels_path = os.path.join(modeldir, 'coco.names')
	weights_path = os.path.join(modeldir, 'yolov3.weights')
	modelconfig_path = os.path.join(modeldir, 'yolov3.cfg')
	trainingfiles = [os.path.join(trainingdir, f) for f in os.listdir(trainingdir)]
	#########


	###########init labels color and network
	labels = open(labels_path).read().strip().split("\n")

	np.random.seed(42)
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

	conf = 0.5
	thresh = 0.3

	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(modelconfig_path, weights_path)
	###############

	#################load an image through the network
	image = cv2.imread(trainingfiles[2])
	(H, W) = image.shape[:2]
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))
	##################


	##############handle drawing bounding box and stuff

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > conf:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thresh)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(60*1000)


main()