# coding=UTF-8
import cv2
import numpy as np
import math


def crop_fn(frame, box, h_scale=1, w_scale=1):

    box = [int(i) for i in box]

    max_h, max_w = frame.shape[:2]
    h, w = box[3] - box[1], box[2] - box[0]

    new_box = [
        max(0, box[0] - w_scale*w),
        max(0, box[1] - h_scale*h),
        min(box[2] + w_scale*w, max_w),
        min(box[3] + h_scale*h, max_h),
    ]

    crop_img = frame[new_box[1] : new_box[3], new_box[0] : new_box[2], :]

    ## 在crop数据当中的位置
    box_in_crop = [
        box[0] - new_box[0],
        box[1] - new_box[1],
        box[2] - new_box[0],
        box[3] - new_box[1],
    ]

    return crop_img, box_in_crop


if __name__ == "__main__":
    import sys
    from plot import plot_one_box

    frame = cv2.imread(
        "/sdc/jjlv/CODE/视频样本挖掘集合/someTestPic/SZ_CH_Q_05_20181002000115-00-00-0066_00000000.jpg"
    )
    box = [0, 500, 100, 600]
    cv2.imwrite("box.jpg", frame[box[1] : box[3], box[0] : box[2], :])
    crop_img, box_in_crop = crop_fn(frame, box)
    plot_one_box(box_in_crop, crop_img)
    cv2.imwrite("box_2.jpg", crop_img)
