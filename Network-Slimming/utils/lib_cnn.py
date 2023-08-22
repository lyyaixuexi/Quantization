# coding=UTF-8
import cv2 
import numpy as np 
import math 

try:
    import sys
    caffe_root = '/data1/wzheng/projects/caffe-quant-seg-sparse-master-bias/'  # Change this to the absolute directory to Caffe
    sys.path.insert(0, caffe_root + 'python')
    import caffe 
except:
    print('YOU ENVIORMENT IS NOT SUPPORT CAFFE AS BACKEND')
try:
    import pycuda
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
except:
    print('YOU ENVIORMENT IS NOT SUPPORT TENSORRT AS BACKEND')


class Preprocess:
    def __init__(
        self,
        autocomplete=None,
        resize=None,
        padding=None,
        scale=None,
        mean=None,
        crop_padding=None,
        color_mode=None,
        yuv_split = None,
        lifting=False,
    ):
        ##这里是专门为了caffemodel格式的预处理,所有的函数应该都为图像输入，返回图像，处理函数需要独立且无序要求的
        self.scale = scale
        # 倍数补齐
        self.autocomplete = autocomplete
        # 图像resize,期望输入是一个元组表示（w,h)
        self.resize = resize
        self.process_list = []
        self.color_mode = color_mode
        self.yuv_split= yuv_split
        self.mean = mean

        if self.resize:
            self.process_list.append(self.Resize)
        if self.autocomplete:
            self.process_list.append(self.AutoComplete)
        ##这里有俩个步骤无法满足完全无序的要求，yuv的处理与否必须放在所有处理之后，且scale必须放在yuv处理之后（因为当前yuv的标准函数输入的模式是一个0~255值域的输入）
        if self.color_mode == 'yuv':
            self.process_list.append(self.Bgr2420sp)
        elif self.color_mode == 'rgb':
            self.process_list.append(self.Bgr2rgb)
        elif self.color_mode == 'gray':
            self.process_list.append(self.Bgr2gray)
        elif self.color_mode == 'bgr':
            pass
        if self.mean:
            self.process_list.append(self.DoMean)
        if self.scale:
            self.process_list.append(self.DoScale)
        ## split must in color mode as yuv
        self.process_list.append(self.ExpendDimensions)
        self.process_list.append(self.Ascontiguousarray)
        if self.yuv_split and self.color_mode == 'yuv':
            self.process_list.append(self.YuvSplit2part)

        #if lifting:
        #    self.process_list.append(self.Lifting)



    def __call__(self, img):
        ##先处理所有的需要的功能函数
        ##然后补齐到4维
        for fun in self.process_list:
            img = fun(img)
        return img
    
    def Lifting(self, img):
        return [img]

    def AutoComplete(self, img):
        h, w = img.shape[0], img.shape[1]
        new_h = int(math.ceil(1.0 * h / self.autocomplete) * self.autocomplete)
        new_w = int(math.ceil(1.0 * w / self.autocomplete) * self.autocomplete)
        ##这里没有想到特别好的方法，因为存在通道数不确定的问题，这里值都为0的矩阵reshape的值仍然都为0
        # 补齐到倍数
        np_ground = cv2.resize(np.zeros_like(img), (new_w, new_h))
        np_ground[:h, :w] = img
        return np_ground

    def Bgr2420sp(self, img):
        yuv420 = cv2.cvtColor(img, cv2.COLOR_BGRA2YUV_I420)
        h, w = yuv420.shape
        y_pos = h // 3 * 2
        uv_length = h // 6
        img = np.zeros(shape=(h, w), dtype=np.uint8)
        u_plane = yuv420[y_pos : y_pos + uv_length, :]
        v_plane = yuv420[y_pos + uv_length :, :]
        img[:y_pos, :] = yuv420[:y_pos, :]
        img[y_pos:, ::2] = np.reshape(u_plane, newshape=(uv_length * 2, w // 2))
        img[y_pos:, 1::2] = np.reshape(v_plane, newshape=(uv_length * 2, w // 2))
        return img

    def Bgr2rgb(self,img):
        img = img[:,:,::-1]
        return img

    def Bgr2gray(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img

    def DoMean(self,img):
        return img-self.mean
    def DoScale(self, img):
        return img / (1.0 * self.scale)

    def Resize(self, img):
        return cv2.resize(img, self.resize)

    def YuvSplit2part(self, img):
        h= img.shape[2]
        return img[:,:,:int(1.0 * h/3*2),:],img[:,:,int(1.0 * h/3*2):,:]

    def ExpendDimensions(self,img):
        if len(img.shape) == 4:
            pass
        elif len(img.shape) == 3:
            img = img.transpose(2, 0, 1)

            img= img[None, ...]
        elif len(img.shape) == 2:
            img= img[None, None, ...]
        return img

    def CropPadding(self, img):
        # TODO 扩方
        return img

    def Padding(self, img):
        # TODO
        return img

    def Ascontiguousarray(self,img):
        return np.ascontiguousarray(img)

class Caffe_model:
    def __init__(self, caffemodel, prototxt,device='cpu'):
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        self.inputlist = [i for i in self.net.inputs.__iter__()]
        self.outputlist = [i for i in self.net.outputs.__iter__()]
        self.device(device)

    def report(self):
        try:
            _ = self._forward_once()
            print("Net forward success")
        except:
            print(
                "Net forward failed! check the caffemodel and prototxt is suit you caffe！"
            )
        input_list = {}
        output_list = {}
        for i in self.inputlist:
            input_list[i] = self.net.blobs[i].data.shape
        for i in self.outputlist:
            output_list[i] = self.net.blobs[i].data.shape
        print("input:{}\noutput{}".format(input_list, output_list))

    def __call__(self, *x):
        assert len(x) == len(self.inputlist)
        for count, data in enumerate(x):
            input_layer_name = self.inputlist[count]
            self.net.blobs[input_layer_name].data[...] = data
        ## the net output is a dict 
        output = self._forward_once()
        return [output[each] for each in self.outputlist]


    def _forward_once(self):
        return self.net.forward()

    def device(self, cuda='cuda'):
        if cuda == 'cuda':
            caffe.set_mode_gpu()
            caffe.set_device(0)

    def reshape(self, *img):
        # 这个reshape是网络输入口的reshape，用来使得网络可以在一个程序当中接受动态大小的输入
        ## reshape by ur input img
        ## 按照顺序来如果有多个输入
        for count, i in enumerate(img):
            input_layer_name = self.inputlist[count]
            n, c, h, w = i.shape
            self.net.blobs[input_layer_name].reshape(n, c, h, w)

class Tensorrt_model:
    '''
    init a tensorrt model
    useage:
    net = Tensorrt_model('ur/engine/file/path')
    output = net(input)
    all of input and output is numpy array
    it is completely suitable that get input data from Preprocess
    '''
    def __init__(self, engine_file):

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()  # 创建context用来执行推断

        ## h指设备端，d指显卡端
        # self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
        # self.h_output1 = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
        # self.h_output2 = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(2)), dtype=trt.nptype(trt.float32))
        # self.h_output3 = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(3)), dtype=trt.nptype(trt.float32))

        # self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        # self.d_output1 = cuda.mem_alloc(1 * self.h_output1.nbytes)
        # self.d_output2 = cuda.mem_alloc(1 * self.h_output2.nbytes)
        # self.d_output3 = cuda.mem_alloc(1 * self.h_output3.nbytes)

        self.h_input = list()
        self.d_input = list()
        self.h_output = list()
        self.d_output = list()
        self.init_h_d_input_output()
        self.stream = cuda.Stream()
        # self.outputs_shape_list = self.get_shape()

    def __call__(self, img):
        return self.forward(img)

    def forward(self, inputlist):
        # start = time.time()
        if not type(inputlist) == list:
            inputlist = list(inputlist)

        for i, x in enumerate(inputlist):
            np.copyto(self.h_input[i], x.astype(trt.nptype(trt.float32)).ravel())
        ##输入每一个输入
        for i in range(len(inputlist)):
            cuda.memcpy_htod_async(self.d_input[i], self.h_input[i], self.stream)

        self.context.execute_async(
            bindings=list(map(int, self.d_input + self.d_output)),
            stream_handle=self.stream.handle,
        )

        ## debug
        # cuda.memcpy_dtoh_async(self.h_output[0], self.d_output[0], self.stream)
        # cuda.memcpy_dtoh_async(self.h_output[1], self.d_output[1], self.stream)
        # cuda.memcpy_dtoh_async(self.h_output[2], self.d_output[2], self.stream)
        # assert False
        ## 取出每一个输出
        ## 从GPU中返回output
        for i in range(len(self.h_output)):
            cuda.memcpy_dtoh_async(self.h_output[i], self.d_output[i], self.stream)
        # 同步流。
        self.stream.synchronize()
        return self.h_output

    def get_shape(self):
        input_shape = list()
        output_shape = list()

        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                input_shape.append(self.engine.get_binding_shape(binding))
            else:
                output_shape.append(self.engine.get_binding_shape(binding))
        return input_shape, output_shape

    def init_h_d_input_output(self):

        inputs_shape_list, outputs_shape_list = self.get_shape()
        print("outputs_shape_list",outputs_shape_list)
        ## init input
        self.h_input = list()
        self.d_input = list()

        for count, temp_shape in enumerate(inputs_shape_list):
            self.h_input.append(
                cuda.pagelocked_empty(
                    trt.volume(temp_shape), dtype=trt.nptype(trt.float32)
                )
            )
            self.d_input.append(cuda.mem_alloc(self.h_input[count].nbytes))

        ## init output
        self.h_output = list()
        self.d_output = list()

        for count, temp_shape in enumerate(outputs_shape_list):
            self.h_output.append(
                cuda.pagelocked_empty(
                    trt.volume(temp_shape), dtype=trt.nptype(trt.float32)
                )
            )
            self.d_output.append(cuda.mem_alloc(self.h_output[count].nbytes))

