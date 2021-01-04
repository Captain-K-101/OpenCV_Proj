import argparse
import cv2
import os
import numpy as np
class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0
	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])
		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H
		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]
	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

# construct the argument parser and parse the arguments


# load our serialized edge detector from disk
protoPath = os.path.sep.join(['hed_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['hed_model', "hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)
# load the input image and grab its dimensions
image = cv2.imread('./mango6.jpg')
(H, W) = image.shape[:2]
# convert the image to grayscale, blur it, and perform Canny
# edge detection
print("[INFO] performing Canny edge detection...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                             mean=(114.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)
# set the blob as the input to the network and perform a forward pass
# to compute the edges
print("[INFO] performing holistically-nested edge detection...")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")
# show the output edge detection results for Canny and
# Holistically-Nested Edge Detection


image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0], dtype="uint8")
upper = np.array([255, 255, 110], dtype="uint8")
mask = cv2.inRange(image1, lower, upper)
mask=~mask
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(mask, cnts, (255,255,255))

image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
canny = cv2.resize(canny, (0, 0), fx = 0.5, fy = 0.5)
hed = cv2.resize(hed, (0, 0), fx = 0.5, fy = 0.5)
mask = cv2.resize(mask, (0, 0), fx = 0.5, fy = 0.5)

mask1 = cv2.bitwise_xor(mask,hed)
mask1[mask == 0] = 255
result=cv2.bitwise_and(image,image,mask=mask1)
result[mask < 255] = 255
ret, mask2 = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)
ret, mask2 = cv2.threshold(mask2, 100, 255, cv2.THRESH_BINARY_INV)
mask2=cv2.cvtColor(mask2,cv2.COLOR_RGB2GRAY)
result[mask2 == 0]=255
cv2.imshow("Input", image)
cv2.imshow("Canny", mask)
cv2.imshow("HED1", hed)
cv2.imshow("mask", mask2)
cv2.imshow("fin", result)
cv2.waitKey(0)
