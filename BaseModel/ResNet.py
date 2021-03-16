"""
@author:Wang Xinsheng
@File:ResNet.py
@description:...
@time:2021-03-15 22:11
"""

import tensorflow as tf
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Layer,MaxPooling2D,ZeroPadding2D
from BaseModule import LayerModule
class Stage(Layer):
    def __init__(self,conv_count,filters,kernels_size,stage_index=-1):
        super(Stage, self).__init__()
        self.sequential = Sequential()
        for index in range(conv_count):
            if index == 0 :
                if stage_index == 1:
                    res = LayerModule.Residual_Layer(identity=False,filters=filters,
                                                     kernels_size=kernels_size,
                                                     strides=[1,1,1]
                                                     )
                else:
                    res = LayerModule.Residual_Layer(identity=False,filters=filters,
                                                     kernels_size=kernels_size,
                                                     strides=[2,1,1]
                                                     )
            else:
                res = LayerModule.Residual_Layer(filters=filters,
                                                 kernels_size=kernels_size)
            self.sequential.add(res)

    def call(self, inputs, **kwargs):
        outputs = self.sequential(inputs)
        return outputs


class ResNet50(Model):
    def __init__(self):
        super(ResNet50, self).__init__()

        # 输入层为偶数，为边缘补0，防止最大池化层出现单数
        self.cba = LayerModule.CBA_Layer(
                                        filter=64,

                                         kernel_size=7,
                                         stride=2)
        self.max_pool = MaxPooling2D(pool_size=(3,3),
                                     strides=(2,2),padding='same')

        self.stage1 = Stage(3,filters=[64,64,256],
                            kernels_size=[1,3,1],stage_index=1)

        self.stage2 = Stage(4,filters=[128,128,512],
                            kernels_size=[1,3,1])
        self.stage3 = Stage(6,filters=[256,256,1024],
                            kernels_size=[1,3,1])
        self.stage4 = Stage(3,filters=[512,512,2048],
                            kernels_size=[1,3,1])


    def build(self, input_shape):
        print('-------------')
        # x = tf.keras.layers.Input(shape=(640, 640, 4))
        # self.call(x)
        super(ResNet50, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        # outputs = self.padding(inputs)
        outputs = self.cba(inputs)
        outputs = self.max_pool(outputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        return  outputs
    

    #
    # def model(self):
    #     x = tf.keras.layers.Input(shape=(640,640,4))
    #     return Model(inputs = x,outputs=self.call(x))

if __name__ == '__main__':
    model = ResNet50()
    model.build(input_shape=(None,640,640,4))
    model.summary()

