import numpy as np
import cv2

def yuv420sp2yuv444(yuv420sp, imgw, imgh):
    yuv444 = np.zeros((imgh, imgw, 3), dtype=np.uint8)
    yuv420sp = np.reshape(yuv420sp,(-1,imgw))
    yuv444[:,:,0] = yuv420sp[:imgh, :]
    uv = yuv420sp[imgh:imgh//2*3, :]
    u = uv[:,::2]
    v = uv[:,1::2]
    
    yuv444[0::2,0::2,1] = u
    yuv444[1::2,0::2,1] = u
    yuv444[0::2,1::2,1] = u
    yuv444[1::2,1::2,1] = u

    yuv444[0::2,0::2,2] = v
    yuv444[1::2,0::2,2] = v
    yuv444[0::2,1::2,2] = v
    yuv444[1::2,1::2,2] = v    
    
    return yuv444

def rgb2yuv444_bt601_video_range(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[0.257,0.504, 0.098],[-0.148, -0.291, 0.439],[0.439, -0.368, -0.071]])
    bias = np.array([16.0, 128.0, 128.0])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.matmul(trans_mat, src_seq) + bias_seq
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst_seq = np.minimum(np.maximum(dst_seq, 0), 255)
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

def rgb2yuv444_bt601_full_range(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[0.299,0.587, 0.114],[-0.168736, -0.331264, 0.500],[0.500, -0.418688, -0.081312]])
    bias = np.array([0.0, 128.0, 128.0])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.matmul(trans_mat, src_seq) + bias_seq
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst_seq = np.minimum(np.maximum(dst_seq, 0), 255)
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3)).astype(np.uint8))
    return dst

def rgb2yuv444_bt709_video_range(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[0.183, 0.614, 0.062],[-0.101, -0.339, 0.439],[0.439, -0.399, -0.040]])
    bias = np.array([16.0, 128.0, 128.0])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq) + bias_seq, 0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

def rgb2yuv444_bt709_full_range(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[0.213,0.715, 0.072],[-0.117, -0.394, 0.511],[0.511, -0.464, -0.047]])
    bias = np.array([0.0, 128.0, 128.0])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq) + bias_seq,0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

def yuv444_bt601_video_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.164, 0.000, 1.596],[1.164, -0.392, -0.813],[1.164, 2.017, 0.000]])
    bias = np.array([16.0, 128.0, 128.0])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.matmul(trans_mat, src_seq-bias_seq)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst_seq = np.minimum(np.maximum(dst_seq, 0), 255)
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    #dst = np.reshape(dst_seq, (img_h, img_w, 3))
    return dst

def yuv444_bt601_full_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.000, 0.000, 1.402],[1.000, -0.344136, -0.714136],[1.000, 1.772, 0.000]])
    bias = np.array([0, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

def yuv444_bt709_video_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.164, 0.000, 1.793],[1.164, -0.213, -0.534],[1.164, 2.112, 0.000]])
    bias = np.array([16, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

def yuv444_bt709_full_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.000, 0.000, 1.540],[1.000, -0.183, -0.459],[1.000, 1.816, 0.000]])
    bias = np.array([0, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst


if __name__ == '__main__':
    # rgb = cv2.imread('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/rgb_ori.jpg')[:,:,::-1]
    # yuv = rgb2yuv444_bt601_video_range(rgb, 640, 352)
    # with open('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/bt601_video_range.yuv', 'wb') as fp:
    #     fp.write(np.ascontiguousarray(yuv))
    # yuv601v = np.fromfile('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/bt601_video_range.yuv',dtype=np.uint8)
    # rgb = yuv444_bt601_video_range2rgb(yuv601v, 640, 352)
    # cv2.imwrite('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/bt601_video_range.jpg', rgb[:,:,::-1])
    
    # rgb = cv2.imread('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/rgb_ori.jpg')[:,:,::-1]
    # yuv = rgb2yuv444_bt709_full_range(rgb, 640, 352)
    # with open('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/bt709_full_range.yuv', 'wb') as fp:
    #     fp.write(np.ascontiguousarray(yuv))
    # yuv601f = np.fromfile('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/bt709_full_range.yuv',dtype=np.uint8)
    # rgb = yuv444_bt601_full_range2rgb(yuv601f, 640, 352)
    # cv2.imwrite('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/bt709_full_range.jpg', rgb[:,:,::-1])

    yuv420sptda4 = np.fromfile('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/tda4_img/test.yuv', dtype=np.uint8)
    yuv444tda4 = yuv420sp2yuv444(yuv420sptda4, 3840, 1080)
    
    rgb_601full = yuv444_bt601_full_range2rgb(yuv444tda4, 3840, 1080)
    cv2.imwrite('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/tda4_601full_range.jpg', rgb_601full[:,:,::-1])

    rgb_709full = yuv444_bt709_full_range2rgb(yuv444tda4, 3840, 1080)
    cv2.imwrite('/home/wzheng/model_zoo_mdc/network_test_case/yuv_color_space/tda4_709full_range.jpg', rgb_709full[:,:,::-1])