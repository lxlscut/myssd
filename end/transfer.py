import tensorflow as tf
import numpy as np

from end.boxutil import compute_iou


def compute_target(default_boxes,gt_boxes,gt_labels,iou_threshold=0.5):
    """ Compute regression and classification targets
        Args:
            default_boxes: tensor (num_default, 4)
                           of format (cx, cy, w, h)
            gt_boxes: tensor (num_gt, 4)
                      of format (xmin, ymin, xmax, ymax)
            gt_labels: tensor (num_gt,)
        Returns:
            gt_confs: classification targets, tensor (num_default,)
            gt_locs: regression targets, tensor (num_default, 4)
        """
    # Convert default boxes to format (xmin, ymin, xmax, ymax)
    # in order to compute overlap with gt boxes
    default_boxes = transform_center_to_corner(default_boxes)
    # iou [num_boxes_a,num_boxes_b]
    iou = compute_iou(default_boxes,gt_boxes)

    # best_default_iou=[num_gt_boxes],每一个gtbox对应的最大的default_boxes的iou
    best_default_iou = tf.math.reduce_max(iou,0)
    # 每一个gtbox对应的最大的default_boxes的编号
    best_default_idx = tf.math.argmax(iou,0)
    # 每一个default_box对应的最佳的iou，对应的gt_box
    best_gt_iou = tf.math.reduce_max(iou,1)
    # 每一个default_box对应的最佳的gt_box的编号
    best_gt_idx = tf.math.argmax(iou,1)
    #TODO 训练时实质就是将所有的default_box送到神经网络中去训练，
    #TODO y值既是每个default_box对应的最佳的gt_box修正梯度以及label
    # tensor_scatter_nd_update函数：按照best_default_idx中的best_gt_idx编号，
    # 找到相应位置，将其中的default_box值替换为best_default_idx.shape[0]的值
    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx,1),
        tf.range(best_default_idx.shape[0],dtype = tf.int64)
    )
    # TODO 最佳的default_box与gt_box的匹配直接将iou置为1
    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx,1),
        tf.ones_like(best_default_idx,dtype = tf.float32)
    )
    # 将每个default对应某个gt_box--->>>每个default_box对应某个label
    gt_conf = tf.gather(gt_labels,best_gt_idx)
    # 必须threshold大于某个值
    # tf.less(a,b):if a<b return true
    # tf.where函数：tf.where(a,b,c),其中a为一个bool向量，b，c与a的形状相同，
    # 返回值为 若a为true 返回相应位置b的位置，否则返回c的相应位置的值
    # gt_conf 为分类信息，小于阈值将其类置为0，大于则按照圆标签来进行分类
    gt_conf = tf.where(tf.less(best_gt_iou,iou_threshold),
                       tf.zeros_like(gt_conf),
                       gt_conf
                       )
    # 每个default_box对应的gt_box坐标集合，待会与每个default_box做回归计算梯度
    gt_boxes = tf.gather(gt_boxes,best_gt_idx)
    gt_locs = encode(default_boxes,gt_boxes)

    return gt_conf,gt_locs



def encode(default_boxes,boxes,variance=[0.1,0.2]):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    boxes = transfrom_coner_to_center(boxes)
    try:
        # locs = tf.concat([(boxes[...,:2]-default_boxes[:,:2])/(default_boxes[:,2:]*variance[0]),
        #              tf.math.log(boxes[...,2:]/default_boxes[:,2:])/variance[1]],axis=-1)
        locs = tf.concat([
            (boxes[..., :2] - default_boxes[:, :2]
             ) / (default_boxes[:, 2:] * variance[0]),
            tf.math.log(boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
            axis=-1)
    except Exception as e:
        print('the error of decode:...'+str(e))

    return locs

def decode(default_boxes,locs,variance=[0.1,0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    boxes = tf.concat([locs[...,:2]*variance[0]*default_boxes[...,:2]+default_boxes[...,:2],
                       tf.math.exp(locs[...,2:]*variance[1])*default_boxes[...,2:]],axis = -1)

    boxes = transform_center_to_corner(boxes)
    return boxes

def transfrom_coner_to_center(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box

def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
            to format (xmin, ymin, xmax, ymax)
        Args:
            boxes: tensor (num_boxes, 4)
                   of format (cx, cy, w, h)
        Returns:
            boxes: tensor (num_boxes, 4)
                   of format (xmin, ymin, xmax, ymax)
        """
    aboxes = tf.concat([(boxes[...,:2]-boxes[...,2:]/2),
                       (boxes[...,:2]+boxes[...,2:]/2)],axis=-1)

    return aboxes

def compute_nms(boxes,scores,nms_threshold,limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    Returns:
        idx: indices of kept boxes
    """
    # indices = []
    # # 先对所有的score进行排序
    # idx = np.argsort(scores)
    # while True:
    #     # 升序排列
    #     # print(idx)
    #     print('idx-shape'+str(idx))
    #     # 计算iou，如果iou大于阈值就将其删除
    #     length = len(idx)
    #     id = length-1
    #     indices.append(idx[id])
    #     # iou[1,len(idx)-1]
    #     boxes_id = np.expand_dims(boxes[idx[id],:],0)
    #     print(boxes_id,boxes[idx[:id],:])
    #     iou = compute_iou(boxes_id,boxes[idx[:id],:])
    #     print('iou-shape'+str(iou))
    #     print('nms'+str(np.where(iou[0]>nms_threshold)[0]))
    #     idx = np.delete(idx,np.concatenate([[id],np.where(iou[0]>nms_threshold)[0]]))
    #     if len(indices)>limit:
    #         break
    # return indices
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = compute_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)


if __name__ == '__main__':
    a = np.array([[0.1,0.1,0.5,0.5],[0.4,0.4,0.5,0.5],[0.2,0.2,0.3,0.3],[0.2,0.2,0.3,0.3]])
    scores = np.array([0.6,0.5,0.3,0.7])
    nms_threshold = 0.3
    limit = 2
    mmm = compute_nms(a,scores,nms_threshold,limit=limit)
    print(mmm)