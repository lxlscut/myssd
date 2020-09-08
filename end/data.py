import tensorflow as tf
import cv2
import yaml
import os
import xml.etree.ElementTree as ET
import numpy as np

from end.data_augmentation import random_patching
from end.getanchor import generate_anchor
from end.transfer import compute_target


class VOCDataset():
    def __init__(self,root_dir,default_boxes,new_size,augmentation=False,num_examples=-1):
        self.Annotations = os.path.join(root_dir,'Annotations')
        self.JPEGImages = os.path.join(root_dir,'JPEGImages')
        # self.image_dir = os.path.join(self.data_dir,'JPEGImages')
        self.ids = list(map(lambda x: x[:-4],os.listdir(self.JPEGImages)))
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        self.name_to_idx = dict([v,k] for k,v in enumerate(self.idx_to_name))
        self.index_path = os.path.join(root_dir,'ImageSets','Layout','trainval.txt')
        self.new_size = new_size
        self.default_boxes = default_boxes
        self.augmentation = augmentation

        if num_examples != -1:
            self.ids = self.ids[:num_examples]

    def get_image(self,index):
        file_name = self.ids[index]
        image_path = os.path.join(self.JPEGImages,file_name+'.jpg')
        img = cv2.imread(image_path)
        return img

    def getannotation(self,index,origin_shape):
        filename = self.ids[index]
        xml_path = os.path.join(self.Annotations,filename+'.xml')
        w,h = origin_shape
        classes = []
        boxes = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        file_name = root.find('filename').text
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        # print(file_name)
        objects = root.findall('object')
        for object in objects:
            # TODO +1,-1的作用是啥
            object_name = object.find('name').text.lower().strip()
            boxloc = object.find('bndbox')
            # print("xmin:"+str(boxloc.find('xmin').text))
            # print("xmax:" + str(boxloc.find('xmax').text))
            # print("ymin:" + str(boxloc.find('ymin').text))
            # print("ymax:" + str(boxloc.find('ymax').text))
            # print("w:" + str(h))
            # print("h:" + str(w))
            xmin = (float(boxloc.find('xmin').text)-1)/h
            xmax = (float(boxloc.find('xmax').text)-1)/h
            ymin = (float(boxloc.find('ymin').text)-1)/w
            ymax = (float(boxloc.find('ymax').text)-1)/w
            classes.append(self.name_to_idx[object_name]+1)
            boxes.append([xmin,ymin,xmax,ymax])

        return np.array(classes,dtype=np.int64),np.array(boxes,dtype=np.float32)
    # 迭代器，迭代产生训练数据
    def generator(self):
        # 先获取所有的index
        # indexes = []
        # path = self.index_path
        # with open(path) as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         indexes.append(line.strip('\n'))
        for index in range(len(self.ids)):
            img = self.get_image(index)
            (w,h,c) = img.shape
            labels,boxes = self.getannotation(index,(w,h))

            # transfer ndarray to tensor
            labels = tf.constant(labels,dtype = tf.int64)
            # print("the class is "+str(labels))
            boxes = tf.constant(boxes,dtype = tf.float32)
            # print("the boxes is ..."+str(boxes))
            # data_augmentation
            if self.augmentation:
                # a = np.random.choice(['flip','patch','orginal'])
                a = 'patch'
                # if a == 'flip':
                #     img,boxes,labels = horizontal_flip(img=img,boxes=boxes,labels=labels)
                if a == 'patch':
                    img,boxes,labels = random_patching(img=img,boxes=boxes,labels=labels)
                # else:
                #     pass
            else:
                pass
            img = cv2.resize(img,(self.new_size,self.new_size),interpolation=cv2.INTER_CUBIC)
            img=(img/127.0)-1.0
            img = tf.constant(img,dtype = tf.float32)

            gt_confs,gt_locs = compute_target(self.default_boxes,boxes,labels)
            # print("运行迭代器")
            yield img,gt_confs,gt_locs

def  create_batch_generator(dir,default_boxes,new_size,batch_size,
                           num_batches,do_shuffle=False,augmentation=False):
    # num_examples = batch_size*num_batches if num_batches>0 else -1
    voc = VOCDataset(root_dir=dir,default_boxes=default_boxes,new_size=new_size,
                     augmentation=augmentation)
    info = {
        'idx_to_name':voc.idx_to_name,
        'name_to_idx':voc.name_to_idx,
    }
    data_set = tf.data.Dataset.from_generator(
        voc.generator,(tf.float32,tf.int64,tf.float32)
    )
    if do_shuffle:
        data_set = data_set.shuffle(40).batch(batch_size)
    else:
        data_set = data_set.batch(batch_size)
    return data_set.take(num_batches),info



if __name__ == '__main__':
    # yml_path = 'E:\code\\regression\ssd\config.yml'

    with open('../config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    default_boxes = generate_anchor('../config.yml')
    batch_generator,info = create_batch_generator(cfg.get('SSD300').get("root_dir"), default_boxes, cfg.get('SSD300').get("image_size"), 32,
                           100, do_shuffle=False, augmentation=False)
    for i, (imgs, gt_confs, gt_locs) in enumerate(batch_generator):
        print(gt_confs[0])
        break




    # yml_path = 'E:\code\\regression\ssd\config.yml'
    # with open('../config.yml') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # default_boxes = generate_anchor(yml_path)
    # batch_size=32
    # num_batches = 10
    # num_examples = batch_size * num_batches if num_batches > 0 else -1
    # voc = VOCDataset(cfg.get('SSD300').get("root_dir"),default_boxes,
    #                  cfg.get('SSD300').get("image_size"),augmentation=False,num_examples=num_examples)
    #
    # a = voc.generator
    # dataset = tf.data.Dataset.from_generator(a,(tf.float32,tf.int64,tf.float32))
    # dataset = dataset.shuffle(40).batch(32)
    # b = dataset.take(32)
    # print(b)
    # for i,(imgs,gt_confs,gt_locs) in enumerate(b):
    #     print(imgs)
