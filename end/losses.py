import tensorflow as tf

# 困难样本挖掘，增强样本的抗误报率
# 将false positive 的样本放入网络重新进行训练
# false positive 真实正常，检测异常
def hard_negative_mining(loss,gt_confs,neg_ratio):
    """
    Args:
    :param loss: (B,num_default)
    :param gt_confs: (B,num_default)
    :param neg_ratio: (negative/positive)
    :return:
    """
    pos_idx = gt_confs>0  #计算出所有不为背景的样本值（True,False）
    num_pos = tf.reduce_sum(tf.cast(pos_idx,dtype = tf.int32),axis =1) #算出数量，并求和
    num_reg = num_pos*neg_ratio #算出负样本的数量

    rank = tf.argsort(loss,axis=1,direction='DESCENDING')
    rank = tf.argsort(rank,axis=1)
    # 按照比例将误差大的负样本提取出来，进行再次重点训练
    neg_idx=rank<tf.expand_dims(num_reg,1)
    # 返回值为bool类型
    return pos_idx,neg_idx

# 误差函数
class SSDlosses(object):
    def __init__(self,neg_ratio,num_class):
        self.neg_ratio = neg_ratio
        self.num_class = num_class
    def __call__(self,confs,locs,gt_confs,gt_locs):
        """
        Args:
        :param confs: 网络输出的分类信息  为[B,num_default,num_class](softmax会输出每个类的概率)
        :param locs:网络输出的位置信息  为【B,num_default,num_class】
        :param gt_confs: 标签值：（b,num_default）
        :param gt_locs:  坐标值： （b,num_default,4）
        :return: conf_loss,loc_loss
        """
        # reduction参数：none:输出一个张量，每个元素均为相应logits的误差
        #               sum；输出的为所有的误差值和
        cross_entropy = tf.keras.losses.\
            SparseCategoricalCrossentropy(from_logits = True,reduction='none')
        print('gt_confs_shape:'+str(gt_confs.shape))
        print('confs_shape:'+str(confs.shape))
        try:
            temp_loss = cross_entropy(gt_confs,confs)
        except Exception as e:
            print("loss:..."+str(e))
        # 困难样本挖掘
        pos_idx,neg_idx = hard_negative_mining(temp_loss,gt_confs,self.neg_ratio)


        cross_entropy = tf.keras.losses.\
            SparseCategoricalCrossentropy(from_logits=True,reduction='sum')
        smooth_l1_loss=tf.keras.losses.Huber(reduction='sum')
        # 交叉熵只在相应的正负样本里面去做
        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx,neg_idx)],
            confs[tf.math.logical_or(pos_idx,neg_idx)])
        # 位置回归只有正样本需要做
        loc_loss = smooth_l1_loss(gt_locs[pos_idx],locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx,dtype = tf.float32))

        conf_loss = conf_loss/num_pos
        loc_loss = loc_loss/num_pos

        return conf_loss,loc_loss


def create_losses(neg_ratio,num_classes):
    criterion = SSDlosses(neg_ratio,num_classes)
    return criterion
