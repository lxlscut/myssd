import numpy as np
import tensorflow as tf
import yaml
import itertools

# input  the path of config file
# the default_boxes generated
def generate_anchor(path):
    with open(path) as f:
        cfg = yaml.load(f)
    ratios = cfg.get('SSD300').get('ratios')
    scales = cfg.get('SSD300').get('scales')
    fm_size =cfg.get('SSD300').get('fm_sizes')
    print(ratios)
    print(scales)
    # 对第k个特征图来做锚框的计算
    a = len(ratios)
    print(a)
    # (cx,cy,w,h)形式
    default_box = []
    for m,f_map in enumerate(fm_size):
        # 首先确定default_box的坐标
        # 特征图为（im_size,im_size）的正方形
        for x,y in itertools.product(range(f_map),repeat=2):
            cx = (x+0.5)/f_map
            cy = (y+0.5)/f_map
        # 计算ratios=1时的坐标
            default_box.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])
            default_box.append([
                cx,
                cy,
                np.sqrt(scales[m]*scales[m+1]),
                np.sqrt(scales[m]*scales[m+1])
            ])
        # 计算ratios不为1时的坐标
            for i in ratios[m]:
                r = np.sqrt(i)
                default_box.append([
                    cx,
                    cy,
                    scales[m]/r,
                    scales[m]*r
                ])
                default_box.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])
    default_box = tf.constant(default_box)
    default_box = tf.clip_by_value(default_box,0.0,1.0)
    return default_box


# if __name__ == '__main__':
#     cfg = generate_anchor('E:\code\\regression\ssd\config.yml')
#     print(cfg.shape)
    # print(cfg)