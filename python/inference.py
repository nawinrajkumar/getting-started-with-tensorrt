from onnx_helper import ONNXClassifierWrapper
import numpy as np
import cv2 as cv
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule



# Setting up Batch size and Precision
BATCH_SIZE=64
PRECISION = np.float32
N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task
trt_model = ONNXClassifierWrapper("resnet_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)

# Generate a dummy batch.
BATCH_SIZE = 32
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = PRECISION)
predictions = trt_model.predict(dummy_input_batch)

vid = cv.VideoCapture(0)

while(True):
    # Capture the video frame
    ret, frame = vid.read()

    # Transferring Frames to GPU
    frame = frame.astype(np.float32)
    frame_gpu = cuda.mem_alloc(frame.nbytes)
    cuda.memcpy_htod(frame_gpu, frame)

    mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
    """)
    # func = mod.get_function("doublify")
    # func(frame_gpu, block=(4,4,1))

    frame_doubled = np.empty_like(frame)
    cuda.memcpy_dtoh(frame_doubled, frame_gpu)
    
    print(frame_doubled.shape)
    cv.imshow('frame', frame)
      
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
