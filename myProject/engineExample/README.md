这个目录下提供了三个 simple 文件，分别是simpleOnnx.cpp、simpleOnnx_1.cpp 和 simpleOnnx_2.cpp

只要有一个编译成功都可以通过：
./simpleOnnx_1 <path_to_model.onnx> <path_to_input.pb> 的方式运行
例如：
./simpleOnnx_1 ../../../unet/unet.onnx  ../../../unet/test_data_set_0/input_0.pb
