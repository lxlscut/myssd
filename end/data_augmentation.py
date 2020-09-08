import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow_core import keras
from PIL import Image
# 功能生成patch框，与boxes的相交阈值
# 值均为（0,1），相对于原图的比例
from end.boxutil import compute_iou


def generate_patch(boxes, threshold):

    while True:
        # 窗宽 随机取得
        patch_w = np.random.uniform(0.1,1)
        # 长宽比例
        ratios = np.random.uniform(0.5,2)
        # 求出相应的长
        patch_h = patch_w*ratios
        # 先算出patch对应的坐标
        patch_xmin = np.random.uniform(0,1-patch_w)
        patch_ymin = np.random.uniform(0,1-patch_h)
        patch_xmax = patch_xmin+patch_w
        patch_ymax = patch_ymin+patch_h
        # todo 此处之所以要使其成为二维 因为如果只用1维会使其在计算iou时发生错误，因为与gt-box不匹配
        patches = np.array([[patch_xmin,patch_ymin,patch_xmax,patch_ymax]],dtype=np.float32)
        # patches范围必须受图片的大小约束
        patches = np.clip(patches,0.0,1.0)
        # 计算iou
        iou = compute_iou(patches,boxes)
        # 如果有一个iou大于threshold，则中断返回该值
        if tf.math.reduce_any(iou>threshold):
            break
    # todo 这里不用iou是因为patch只有一行，是【【********】】
    return iou[0],patches[0]



# 功能：在原始图像中随机选取一个子框来进行操作
def random_patching(img, boxes, labels):
    # 随机取一个阈值
    threshold = np.random.choice([0.3,0.5,0.7])
    ious,patch = generate_patch(boxes=boxes,threshold=threshold)
    patch_w = patch[2]-patch[0]
    patch_h = patch[3]-patch[1]
    # 首先判断ious,patch是否可用
    # todo 为什么判断boxes的中心？？？
    boxes_center = (boxes[:,:2]+boxes[:,2:])/2
    # 这里运算的其实是一个bool数组
    useful = ((ious>0.3)&(boxes_center[:,0]>patch[0])&(boxes_center[:,0]<patch[2])
              &(boxes_center[:,1]>patch[1])&(boxes_center[:,1]<patch[3]))
    # 不满足条件直接返回原图，不做处理
    if not tf.math.reduce_any(useful):
        return img,boxes,labels
    boxes = boxes[useful]
    boxes = tf.stack([(boxes[:,0]-patch[0])/patch_w,
                      (patch[1]-boxes[:,1])/patch_h,
                      (patch[2] - boxes[:, 2]) / patch_w,
                      (boxes[:,3]-patch[3])/patch_h],axis = 1)
    boxes = np.clip(boxes,0.0,1.0)
    labels = labels[useful]
    # img = img.crop(patch)
    print("patch.....",str(patch))
    img = img[int(patch[0]*img.shape[0]):int(patch[2]*img.shape[0]),int(patch[1]*img.shape[1]):int(patch[3]*img.shape[1])]
    print("img_shape..."+str(img.shape))
    return img,boxes,labels


# 功能：实现图像的翻转等操作
def horizontal_flip(img, boxes, labels):
    a = np.random.choice([0,1])
    # 等于1时水平翻转
    # 等于0时竖直翻转,随机翻转
    img = cv2.flip(img,flipCode=a)
    if a==1:
        # 按列堆叠在一起,boxes为多维数据，不能直接boxes[0]
        boxes = tf.stack([1-boxes[:,2],boxes[:,1],1-boxes[:,0],boxes[:,3]],axis = 1)
    elif a==0:
        boxes = tf.stack([boxes[:,0],1-boxes[:,3],boxes[:,2],1-boxes[:,1]],axis = 1)
    else:
        raise ValueError
    return img,boxes,labels


if __name__ == '__main__':
    img = cv2.imread('E:\code\\regression\ssd\dataset\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\\000005.jpg')
    print(img.shape)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    img = horizontal_flip(img)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    print(img.shape)