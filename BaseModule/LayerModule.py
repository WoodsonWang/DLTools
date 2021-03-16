"""
@author:Wang Xinsheng
@File:LayerModule.py
@description:...
@time:2021-03-15 22:21
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,BatchNormalization,ReLU,add

from tensorflow.keras import Sequential

class CBA_Layer(Layer):
    '''
    定义卷积，批归一化，激活函数层，cba
    '''
    def __init__(self,filter,kernel_size,stride=1,a_flag=True):
        '''
        :param a_flag 是否需要激活函数
        '''
        super(CBA_Layer, self).__init__()
        self.a_flag = a_flag
        self.conv = Conv2D(filter,
                           kernel_size=(kernel_size,kernel_size),
                           strides=(stride,stride),
                           padding='same'

                           )
        self.bn = BatchNormalization()
        self.activation = ReLU()

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        if self.a_flag:
            outputs = self.activation(outputs)
        return outputs

class Residual_Layer(Layer):
    '''
    定义残差层
    '''
    def __init__(self,filters=[64,64,64],kernels_size=[3,3,3],strides=[1,1,1],identity=True):
        super(Residual_Layer, self).__init__()
        self.identity = identity
        self.sequential = Sequential()
        for index in range(len(filters)):
            if index == len(filters)-1:
                conv = CBA_Layer(filters[index],kernels_size[index],strides[index],a_flag=False)
            else:
                conv = CBA_Layer(filters[index],kernels_size[index],strides[index])
            self.sequential.add(conv)
        self.short_cut = CBA_Layer(filter=filters[-1],
                                   kernel_size=1,
                                   stride=strides[0],
                                   a_flag=False
                                   )
        self.activation = ReLU()
    def call(self, inputs, **kwargs):
        outputs_one = self.sequential(inputs)
        if self.identity:
            outputs = add([inputs,outputs_one])
        else:
            outputs_two = self.short_cut(inputs)
            outputs = add([outputs_one,outputs_two])
        outputs = self.activation(outputs)
        return outputs

if __name__ == '__main__':
    from tensorflow.keras.applications import ResNet50
    model = ResNet50()
    model.summary()
