"""
@author:Wang Xinsheng
@File:SegNet.py
@description:...
@time:2021-03-05 16:37
"""

from tensorflow.keras import Model
import tensorflow as tf

class Decoder_Block(tf.keras.layers.Layer):
    def __init__(self,conv_count,filters):
        super(Decoder_Block, self).__init__()
        self.unsample = tf.keras.layers.UpSampling2D(
            size=(2,2)
        )
        self.repeat_layer = tf.keras.Sequential()
        for i in range(conv_count):
            conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(3,3),
                padding='same',
                name='block1_conv1'
            )
            bn = tf.keras.layers.BatchNormalization()
            activation = tf.keras.layers.Activation(tf.nn.relu)

            self.repeat_layer.add(conv)
            self.repeat_layer.add(bn)
            self.repeat_layer.add(activation)

    def call(self, inputs, **kwargs):
        out = self.unsample(inputs)
        out = self.repeat_layer(out)
        return out

class Encoder_Block(tf.keras.layers.Layer):
    def __init__(self,conv_count,filters):
        super(Encoder_Block, self).__init__()
        self.repeat_layer = tf.keras.Sequential()
        for i in range(conv_count):
            conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(3,3),
                padding='same',
                name='block1_conv1'
            )
            bn = tf.keras.layers.BatchNormalization()
            activation = tf.keras.layers.Activation(tf.nn.relu)

            self.repeat_layer.add(conv)
            self.repeat_layer.add(bn)
            self.repeat_layer.add(activation)
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                                    strides=(2,2),
                                                    name='block1_pool')

    def call(self, inputs, **kwargs):
        out = self.repeat_layer(inputs)
        out = self.pooling(out)
        return out

class Encoder_Layer(tf.keras.layers.Layer):
    '''
    编码层
    '''
    def __init__(self,params):
        super(Encoder_Layer, self).__init__()
        self.repeat = tf.keras.Sequential()
        for param in params:
            self.repeat.add(Encoder_Block(param[0],param[1]))



    def call(self, inputs, **kwargs):
        out = self.repeat(inputs)
        return out

class Decoder_Layer(tf.keras.layers.Layer):
    '''
    解码层
    '''
    def __init__(self,params):
        super(Decoder_Layer, self).__init__()
        self.repeat = tf.keras.Sequential()
        for param in params:
            self.repeat.add(Decoder_Block(param[0],param[1]))

    def call(self, inputs, **kwargs):
        out = self.repeat(inputs)
        return out


class segnet(Model):
    def __init__(self):
        super(segnet, self).__init__()
        params = [(2,64),(2,128),(3,256),(3,512),(3,512)]
        self.encoder = Encoder_Layer(params)
        self.decoder = Decoder_Layer(params[::-1])

        self.conv = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=(1,1),
                                           strides=(1,1),padding='same')
        self.activation = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        out = self.encoder(inputs)
        out = self.decoder(out)
        out = self.conv(out)
        out = self.activation(out)
        return out

    def model(self):
        x = tf.keras.Input(shape=(640,640,4))
        return Model(inputs = x, outputs = self.call(x))

if __name__ == '__main__':
    model = segnet().model()
    print(model.summary())