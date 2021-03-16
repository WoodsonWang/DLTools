"""
@author:Wang Xinsheng
@File:PSPNet.py
@description:...
@time:2021-03-15 20:44
"""
import tensorflow as tf
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Layer,AveragePooling2D,add,\
    Lambda,concatenate,Dropout,Conv2D,Activation,Input
from BaseModel import ResNet
from BaseModule import LayerModule
from tensorflow.keras.applications import ResNet50


class Pool(Layer):
    def __init__(self,pool_size,stride):
        super(Pool, self).__init__()
        self.average_pool = AveragePooling2D(pool_size=(pool_size,pool_size),
                                             strides=(stride,stride),
                                             padding='same')

    def call(self, inputs, **kwargs):
        outputs = self.average_pool(inputs)
        return outputs


class PPM(Layer):
    def __init__(self,ratio=[1,2,3,6],**kwargs):
        super(PPM, self).__init__(**kwargs)
        print('PPM1===============================')
        self.ratio = ratio

    def build(self, input_shape):
        '''
        在build层进行动态初始的参数设置
        '''
        print('PPM2===============================')
        _,h,w,c = input_shape
        self.pool_list = []
        # 经过池化后组合的通道数和输入的通道数相同，将通道数平分给不同的池化层
        filter = c // len(self.ratio)
        for ratio in self.ratio:


            pool_size = stride_size = h/ratio
            average_pool = AveragePooling2D(pool_size=(pool_size,pool_size),strides=(stride_size,stride_size),padding='same')
            # 使用1*1 卷积对通道数进行调整
            conv = LayerModule.CBA_Layer(filter=filter,
                                         kernel_size=1)
            # 使用lambda定义上采样，尺寸为输入的特征图的大小
            resize = Lambda(lambda x:tf.image.resize(x,(h,w)))
            sequential = Sequential([average_pool,
                                    conv,
                                    resize])
            self.pool_list.append(sequential)


    def call(self, inputs, **kwargs):
        print('PPM3===============================')
        outputs = [inputs]
        for pool in self.pool_list:
            outputs.append(pool(inputs))
        # 将不同池化的结果,以及输入的原始特征图进行堆叠
        outputs = concatenate(outputs)
        print(outputs.shape)
        return outputs


class PSPNet(Model):
    def __init__(self):
        super(PSPNet, self).__init__()

        self.resnet = ResNet.ResNet50()
        self.ppm = PPM()
        self.cba = LayerModule.CBA_Layer(filter=64,
                                         kernel_size=3,
                                         )
        # 丢弃10%
        self.dropout = Dropout(rate=0.1)
        self.conv = Conv2D(filters=1,
                           kernel_size=(1,1),
                           strides=(1,1),
                           padding='same')
        self.activation = Activation(tf.nn.sigmoid)





    def build(self, input_shape):
        print('1=========================')

        _,h,w,c = input_shape
        self.resize = Lambda(lambda x:tf.image.resize(x,(h,w)))

        x = tf.keras.layers.Input(shape=(640, 640, 4))
        self.call(x)

        super(PSPNet, self).build(input_shape)



    def call(self, inputs, training=None, mask=None):
        print('3=========================')
        print(inputs.shape)
        outputs = self.resnet(inputs)
        outputs = self.ppm(outputs)
        outputs = self.cba(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv(outputs)
        outputs = self.resize(outputs)
        outputs = self.activation(outputs)
        return outputs




if __name__ == '__main__':
    model = PSPNet()
    model.build(input_shape=(None,640,640,4))
    model.summary()