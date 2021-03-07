"""
@author:Wang Xinsheng
@File:model.py
@description:...
@time:2020-12-30 16:45
"""

import tensorflow as tf
from tensorflow.keras import Model



class DepthwiseSeparableLayer(tf.keras.layers.Layer):
    '''

    自定义的深度可分离卷积层
    可以决定卷积之间是否添加激活函数
    '''
    def __init__(self,filters,strides=(1,1),rate=1):
        super(DepthwiseSeparableLayer, self).__init__()

        self.depth = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3),
                                            strides=strides,
                                            dilation_rate=(rate,rate),
                                            padding='same',
                                            use_bias=False,
                                            )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.point = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=(1,1),
                                        strides=(1,1),
                                        padding='same',
                                        use_bias=False)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(tf.nn.relu)

    def call(self, inputs, **kwargs):
        output = self.depth(inputs)
        output = self.bn1(output)
        output = self.point(output)
        output = self.bn2(output)
        output = self.relu(output)
        return output


class bilinear_upsampling(tf.keras.layers.Layer):
    def __init__(self):
        '''
        修改图像尺寸
        '''
        super(bilinear_upsampling, self).__init__()
        self.resize = tf.image.resize

    def call(self, inputs, **kwargs):
        from tensorflow.python.keras.utils import conv_utils
        shape = kwargs['shape']
        out  = tf.image.resize(inputs,
                               shape[1:3],
                               method='bilinear',
                               antialias=True)
        return out

class repeat_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(repeat_layer, self).__init__()
        self.sep1 = DepthwiseSeparableLayer(728,strides=(1,1))
        self.sep2 = DepthwiseSeparableLayer(728,strides=(1,1))
        self.sep3 = DepthwiseSeparableLayer(728,strides=(1,1))
        # 残差结构


    def call(self, inputs, **kwargs):
        output = self.sep1(inputs)
        output = self.sep2(output)
        output = self.sep3(output)
        # 残差相加
        output = tf.keras.layers.add([output,inputs])
        return output


class xception_block(tf.keras.layers.Layer):
    '''
    定义自己的层
    '''
    def __init__(self,feature_count_list,skip_connection_type='conv',skip_out = False):
        super(xception_block, self).__init__()
        self.skip_out = skip_out
        # 首先对每个通道上的特征独立执行卷积操作，其次对整体进行1*1卷积操作.
        self.depthconv1 = DepthwiseSeparableLayer(feature_count_list[0])
        self.depthconv2 = DepthwiseSeparableLayer(feature_count_list[1])
        self.depthconv3 = DepthwiseSeparableLayer(feature_count_list[2],strides=(2,2))

        # 跳层结构
        if skip_connection_type == 'conv':
            # 卷积残差网络
            # 用来调整x维度的，使得跟输出的层的特征图数目一样
            self.downsample = tf.keras.Sequential()
            self.downsample.add(
                tf.keras.layers.Conv2D(filters=feature_count_list[-1],
                                       kernel_size=(1,1),
                                       strides=(2,2),
                                       padding='valid')
            )
            self.downsample.add(
                tf.keras.layers.BatchNormalization()
            )
        else:
            # def g(x): return x
            self.downsample = lambda x:x



    def call(self, inputs, **kwargs):
        out = self.depthconv1(inputs)
        skip_output = self.depthconv2(out)
        out = self.depthconv3(skip_output)
        identity = self.downsample(inputs)
        output = tf.keras.layers.add([out,identity])


        if not self.skip_out:
            return output
        else:
            return output,skip_output

class GlobalAveragePooling(tf.keras.layers.Layer):
    '''
    自定义池化上采样层
    '''
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        # 扩展维度的函数
        self.expand = lambda x: tf.expand_dims(x, axis=1)
        # 增加特征图，跟其他空洞卷积的结果一致
        self.conv = tf.keras.layers.Conv2D(filters=256,
                                           kernel_size=(1,1),
                                           strides=(1,1),
                                           padding='same',
                                           use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(tf.nn.relu)
        self.resize = bilinear_upsampling()


    def call(self, inputs, **kwargs):
        shape = inputs.shape
        output = self.pooling(inputs)
        output = self.expand(output)
        output = self.expand(output)
        output = self.conv(output)
        # output = self.bn(output)
        output = self.relu(output)
        output = self.resize(output,shape=shape)
        return output

class ASPPLayer(tf.keras.layers.Layer):
    '''
    使用不同倍率的空洞卷积
    '''
    def __init__(self):
        super(ASPPLayer, self).__init__()
        # 1*1 卷积
        self.conv1_1 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=(1,1),
                                              strides=(1,1),
                                              padding='same',
                                              use_bias=False)

        # 3*3 rate = 6
        self.conv3_1 = DepthwiseSeparableLayer(filters=256,
                                               rate=6
                                               )

        # 3*3 rate = 12
        self.conv3_2 = DepthwiseSeparableLayer(filters=256,
                                               rate=12
                                               )
        # 3*3 rate = 18
        self.conv3_3 = DepthwiseSeparableLayer(filters=256,
                                               rate=18
                                               )
        # 池化层
        self.pooling = GlobalAveragePooling()

        self.concatenate = tf.keras.layers.Concatenate()


    def call(self, inputs, **kwargs):
        y1 = self.conv1_1(inputs)
        y2 = self.conv3_1(inputs)
        y3 = self.conv3_2(inputs)
        y4 = self.conv3_3(inputs)
        y5 = self.pooling(inputs)
        outputs = self.concatenate([y1,y2,y3,y4,y5])
        return outputs

class DecoderLayer(tf.keras.layers.Layer):
    '''
    自定义解码层
    '''
    def __init__(self):
        super(DecoderLayer, self).__init__()
        # 上采样层
        self.upsample = bilinear_upsampling()
        # 卷积 1*1 卷积第一个
        self.conv1_1 = tf.keras.layers.Conv2D(filters=48,
                                              kernel_size=(1,1),
                                              strides=(1,1),
                                              padding='same',
                                              use_bias=False)
        self.conv1_1_bn = tf.keras.layers.BatchNormalization()
        self.conv1_1_relu = tf.keras.layers.Activation(tf.nn.relu)

        self.concatenate = tf.keras.layers.Concatenate()
        self.sep1 = DepthwiseSeparableLayer(filters=256)
        self.sep2 = DepthwiseSeparableLayer(filters=256)


    def call(self, inputs, **kwargs):
        # 获得跳层输入
        skip_inputs = kwargs['skip_inputs']
        shape = skip_inputs.shape
        output_encoder = self.upsample(inputs,shape=shape)

        output_low_level = self.conv1_1(skip_inputs)
        output_low_level = self.conv1_1_bn(output_low_level)
        output_low_level = self.conv1_1_relu(output_low_level)

        outputs = self.concatenate([output_encoder,output_low_level])
        outputs = self.sep1(outputs)
        outputs = self.sep2(outputs)
        return outputs


class CBA_layer(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides,drop=False,**kwargs):
        '''''

        :param strides: 步长
        :param filters: 模板数
        '''''
        super(CBA_layer, self).__init__(**kwargs)
        self.drop = drop

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            dilation_rate=(1,1)
            ,)
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Activation1 = tf.keras.layers.Activation(tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.BN1(out)
        out = self.Activation1(out)
        if self.drop:
            out = self.dropout(out)
        return out



class DeeplabV3plus(Model):
    def __init__(self):
        '''
        定义网络结构
        '''
        super(DeeplabV3plus,self).__init__()
        # 第一个卷积层 dilation_rate=(1,1) 是默认值
        # entry flow
        self.conv1 = CBA_layer(32,kernel_size=(3,3),strides=(2,2),name='MyLayer_D1')
        self.conv2 = CBA_layer(64,kernel_size=(3,3),strides=(1,1))


        self.res1 = xception_block([32,32,32])
        # 多一个返回，作为浅层特征
        self.res2 = xception_block([256,256,256],skip_out=True)

        self.res3 = xception_block([728,728,728])

        # 循环16次的结构
        # middle flow
        self.res_repeat = tf.keras.Sequential()
        for i in range(16):
            self.res_repeat.add(repeat_layer())

        # exit flow
        self.res4 = xception_block([728,728,1024])

        self.sep1 = DepthwiseSeparableLayer(filters=1536)
        self.sep2 = DepthwiseSeparableLayer(filters=1536)
        self.sep3 = DepthwiseSeparableLayer(filters=2048)

        self.aspp = ASPPLayer()
        # 1*1 卷积
        self.conv3 = CBA_layer(filters=256,kernel_size=(1,1),strides=(1,1),drop=True)

        # decoder
        self.decoder = DecoderLayer()

        # 需要分几个类就使用几个filters
        self.conv4 = tf.keras.layers.Conv2D(filters=1,
                                            kernel_size=(1,1),
                                            strides=(1,1),
                                            padding='same')
        self.upsample = bilinear_upsampling()
        # 使用softmax 获取每一个像素点的类别
        # 对每一个像素点的多个波段求softmax结果
        self.activation =tf.keras.layers.Activation(tf.nn.sigmoid)

        #执行build函数来确定输入形状 不调用无法输出形状，或者调用model()函数也可以打印形状
        # self.build(input_shape=(None,640, 640, 3))




    def call(self, inputs, training=None, mask=None):
        '''
        调用网络结构块，实现向前传播
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.conv1(inputs)
        self.shape = inputs.shape
        x = self.conv2(x)
        x = self.res1(x)
        x,skip_output = self.res2(x)
        x = self.res3(x)
        x = self.res_repeat(x)
        x = self.res4(x)
        x = self.sep1(x)
        x = self.sep2(x)
        x = self.sep3(x)
        x = self.aspp(x)
        x = self.conv3(x)
        x = self.decoder(x,skip_inputs = skip_output)
        x = self.conv4(x)
        x = self.upsample(x,shape=self.shape)

        x = self.activation(x)
        return x



    def model(self):
        # 调用函数，显示结构和参数 否则显示Multiple
        x = tf.keras.layers.Input(shape=(640, 640, 4))
        return Model(inputs=x,outputs=self.call(x))






def initSettings():
    # 占位符会传入监控指标
    checkpoint_path = './checkpoint/berlin.{epoch:02d}-{val_loss:.4f}.H5'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           verbose=0,
                                           save_freq='epoch'),
        # 终止训练的回调函数
        # 监控val_loss ,patience指连续3次越来越差就不再训练了
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3),
    ]
    return callbacks

from dataset_factory import *
from dataTool import *



if __name__ == '__main__':
    import os
    # os.environ["PATH"] += os.pathsep + 'C:\Program Files\graphviz-2.44.2~dev.20201112.1525-win32\Graphviz\\bin'  #注意修改你的路径

    os.environ['CUDA_VISIBLE_DEVICES'] = '/cpu:0'
    # with tf.device('/gpu:1'):
    # tf.config.set_soft_device_placement(True)
    train_data_path,val_data_path = get_config()
    # train_dataset =read_TFRecord_file(train_data_path)
    # train_dataset = train_dataset.shuffle(buffer_size=1600)
    # train_dataset = train_dataset.batch(batch_size=2)
    val_dataset = read_TFRecord_file(val_data_path)
    # val_dataset 被封装成一个[1,80,640,640,4]的数据集
    val_dataset = val_dataset.batch(batch_size=80)
    # 取出第一个batch
    val_dataset = val_dataset.take(1)
    count= 0
    # 把一个batch中的每个元素数据取出来，
    for elements in val_dataset.unbatch():
        print(elements)
        count += 1
    print(count)
    # print(val_dataset)
    # for i in val_dataset:
    #     count+=1
    # print(count)
    # print(train_dataset)



    # model = DeeplabV3plus().model()
    # model.build(input_shape=(None,640, 640, 3))
    # # # 打印模型结构
    # print(model.summary())
    # tf.keras.utils.plot_model(model,'deeplabv3+.png',show_shapes=True,show_layer_names=True)
#-----------------------------------------------------------------
# 执行模型训练
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1,
                                                                   decay_steps=1,
                                                                   decay_rate=0.96)
    # train_epochs = 100
    # model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    #               loss= tf.keras.losses.binary_crossentropy, # 损失函数
    #               metrics=['accuracy'])
    # # #
    # train_history = model.fit(train_dataset,
    #                           validation_data=val_dataset,
    #                           epochs=train_epochs,
    #                           verbose=2,# 每个epoch输出一行记录
    #                           )
#-----------------------------------------------------------------

# 执行模型预测
    import numpy as np
    import matplotlib.pyplot as plt
    weights_path = r'D:\codeproject\python\DeepLearning\SegmentationNetwork\weights\sgd\yumi.51-0.2625.H5'
    #
    # 告诉模型输入的参数格式
    # model.build(input_shape=(None,64000, 64000,4))

    # image = image[...,0:3]
    # print(image.shape)
    # plt.imshow()
    #
    # plt.show()

    #
    # model = DeeplabV3plus()
    # image = cv2.imread(r'F:\lishu\1\traindata\img\3.tif',cv2.IMREAD_UNCHANGED)
    # print(image)
    # # submodel = tf.keras.models.Model(inputs = model.input,outputs = model.layers[-2].output)
    # model.load_weights(weights_path)
    # image = image.reshape((1,640, 640, 4))
    # image = tf.cast(image, tf.float32)/(1.0*255)
    # # print(image.numpy)
    # y = model.predict(image)
    # print(y)
    # print(y.shape)
    # z = np.zeros((640,640))
    # z[y[0,:,:,0]>=0.5] = 1
    #
    # cv2.imwrite(r'F:\lishu\1\traindata\7.tif',z)
    # print(z)

#-----------------------------------------------------------------