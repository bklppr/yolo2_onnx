
##Problems to be solved

1. Image Classification, only VGG series can be coverted sucessfully. Other pytorch pre-train model like Alexnet, DenseNet... suffer from failed convetion or inference error. And VGG score need to be double-comfirmed.


2. Although there are "GPU:0" and "CPU" two kinds of device to be choose, but the time costs are similar(42s and 43s), need to be double-comfirmed.

3. 


##Notification

1. yolo出來的預測圖之所以是方形而非原本的長方形，是因為圖像進去時有被resize成方形
2. （Problems補充）Alexnet用onnx_caffe2會成功而onnx_tf因某些操作不支援而失敗, 有些模形無論用onnx_caffe2或onnx_tf都轉不成功，有些就算轉成功了也無法成功用來預測
3. 因onnx IR沒有每一層的input/output shape, 須自行計算, 或可考慮轉換至其他框架，以其他框架的api計算

