# yolov5_tensorrt_mini
最小化的yolov5 tensorrt部署

系统 ubuntu18.04
tenorrt版本>8.0

使用方法：

git clone https://github.com/ultralytics/yolov5

git clone https://github.com/doorteeth/yolov5_tensorrt_mini.git

$ cd path/to/yolov5

$ python path/to/export.py --weights yolov5s.pt --include onnx

生成yolov5s.onnx,将文件复制到yolov5_tensorrt_mini项目下

$ cp yolov5s.onnx path/to/yolov5_tensorrt_mini/

$ cd path/to/yolov5_tensorrt_mini

$ trtexec --onnx=yolov5s.onnx --saveEngine=test.engine

$ mkdir build

$ cmake ..

$ make -j8

$ ./yolov5mini_tensorrt

相关视频：https://www.bilibili.com/video/BV1c34y1B7P5?spm_id_from=333.999.0.0
