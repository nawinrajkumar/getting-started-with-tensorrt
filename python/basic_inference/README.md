# Deploying ResNet Onnx model onto Tensorrt

This example shows how to deploy a custom model onto tensorrt by converting the model onto ONNX. For this example we have used ResNet pretrained model from Pytorch.

Note the following commands are used in Ubuntu. And it is highly recommended to use the following on Ubuntu 20.04 or Ubuntu 22.04.:

Run the following python **main.py** using the following command:
```
python main.py
```

> **Note:** If the model import gets killed while running the program, reduce the BATCH_SIZE in the program.  

After running the program you will have a model file named **resnet50_onnx_model**. Now we'll have to build an engine using the **trtexec** command.

```
trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt
```

This will convert our resnet50/model.onnx to a TensorRT engine named resnet_engine.trt.

**Note:**

* To tell trtexec where to find our ONNX model, run:
```--onnx=resnet50/model.onnx``` 

* To tell trtexec where to save our optimized TensorRT engine, run:
```--saveEngine=resnet_engine_intro.trt``` 

If you are facing an error: 

> 'trtexec' command not found.

Use the following command:

```
alias trtexec="/usr/src/tensorrt/bin/trtexec"
```
