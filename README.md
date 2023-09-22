# Framework
<div align=center>
<img src="https://github.com/liujf69/YoloV5-Server/blob/main/framework.png"/>
</div>

# Project Description
1. Build tcp server and tcp client based on C/S model. <br />
2. Use Epoll to monitor the socket connections between server and client. <br />
3. The C/S use the same protocol to send and receive data, which uses four bytes to represent the width, height and size of images. <br />
4. Create two reading threads to receive the data sent by clients and use the least connections algorithm for load balance. <br />
5. Build a thread pool and the data recv by two reading threads will be sent to the worker thread for processing. <br />
6. The worker thread loads the YoloV5 engine for inference based on TensorRT. <br />

# 中文描述
1. 基于 C/S 模型来构建 TCP 服务器和 TCP 客户端。<br />
2. 使用 Epoll 来监控服务器和客户端之间的连接。<br />
3. 服务器和客户端约定使用相同的数据协议，头部分别使用 4 个字节来表示图片的宽，高和大小。<br />
4. 创建两个读线程来接收客户端的数据，并采用 least connections 算法来实现两个读线程的负载均衡。<br />
5. 创建一个可变参数的线程池，读线程接收数据后将图片传入工作线程中，工作线程对图片进行模型推理。<br />
6. 工作线程基于 TensorRT 推理框架来加载 YoloV5 推理引擎进行模型推理，并将推理结果发送回客户端。<br />

# Prerequisites
You must install dependencies like ```Opencv, TensorRT, Cuda``` based on manual compilation. <br />
You must refer to Project [TensorRT-Demo](https://github.com/liujf69/TensorRT-Demo/tree/master/TRT_YoloV5) to serialize your own inference engine locally. <br />

# Build and Run
```
mkdir build && cd build
cmake ..
make
./server 9190
./client 127.0.0.1 9190
```

# Test
<div align=center>
<img src="https://github.com/liujf69/YoloV5-Server/blob/main/test_server.png"/>
</div>
<div align=center>
<img src="https://github.com/liujf69/YoloV5-Server/blob/main/show.png"/>
</div>

# Contact
For any questions, feel free to contact: ```liujf69@mail2.sysu.edu.cn```
