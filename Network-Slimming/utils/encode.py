import cv2
import base64
import numpy as np

def np2base64(npimg):
    img_str = cv2.imencode('.jpg', npimg)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
    b64_code = base64.b64encode(img_str)
    data = str(b64_code, 'utf-8')
    return data

def base64ToNp(image_base64):
    decodeImg = base64.b64decode(image_base64)
    basenp = np.frombuffer(decodeImg, dtype=np.uint8)
    frame = cv2.imdecode(basenp, cv2.IMREAD_COLOR)
    return frame
