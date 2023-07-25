from onnx_helper import ONNXClassifierWrapper
import numpy as np
import cv2 as cv

BATCH_SIZE=64

PRECISION = np.float32
N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task
trt_model = ONNXClassifierWrapper("resnet_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)

# Generate a dummy batch.
BATCH_SIZE = 32
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = PRECISION)

predictions = trt_model.predict(dummy_input_batch)

cv.imshow("predictions", predictions)
cv.waitKey(0)
