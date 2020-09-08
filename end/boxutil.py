import tensorflow as tf
import numpy as np

# boxes_a = [num_boxes,xmin,ymin,xmax,ymax]
# boxes_b = [num_boxes,xmin,ymin,xmax,ymax]
# return overlap[num_boxes_a,num_boxes_b]
def compute_iou(boxes_a,boxes_b):
    # boxes_a = [1,num_boxes,xmin,ymin,xmax,ymax]
    boxes_a = tf.expand_dims(boxes_a,axis = 1)
    # boxes_b = [num_boxes,1,xmin,ymin,xmax,ymax]
    boxes_b = tf.expand_dims(boxes_b,axis = 0)
    # 最大的xmin,ymin
    min_side = tf.maximum(boxes_a[...,:2],boxes_b[...,:2])
    # 最小的xmax,ymax
    max_side = tf.minimum(boxes_a[...,2:],boxes_b[...,2:])

    inter_area = compute_area(min_side,max_side)
    # inter_area1 = (max_side[...,0]-min_side[...,0])*(max_side[...,1]-min_side[...,1])
    # area_a1 = (boxes_a[...,2]-boxes_a[...,0])*(boxes_a[...,3]-boxes_a[...,1])
    # area_b1 = (boxes_b[...,2]-boxes_b[...,0])*(boxes_b[...,3]-boxes_b[...,1])
    # print("inter_area1.shape:"+str(inter_area1.shape))
    # union = area_a1+area_b1
    # print(print("union.shape:"+str(union.shape)))
    area_a = compute_area(boxes_a[...,:2],boxes_a[...,2:])
    # print(print("area_a.shape:" + str(area_a.shape)))
    area_b = compute_area(boxes_b[...,:2],boxes_b[...,2:])
    # print(print("area_b.shape:" + str(area_b.shape)))
    overlap = inter_area/(area_a+area_b-inter_area)

    return overlap

def compute_area(top_left,down_right):
    # todo 大小规定在0.0--512.0 图片的大小范围在0-512
    wh = tf.clip_by_value(down_right-top_left,0.0,512.0)
    area = wh[...,0]*wh[...,1]
    return area


if __name__ == '__main__':
    # [3,4]
    a = np.array([[1,1,2,2]],dtype=np.float32)
    # [4,4]
    b = np.array([[1,1,2,2],[2,2,3,3],[3,3,4,4],[1,1,4,4]],dtype=np.float32)
    overlap = compute_iou(a,b)
    print(overlap[0])
