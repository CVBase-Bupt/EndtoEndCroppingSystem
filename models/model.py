from keras.models import Model
from keras.layers import (
    Input, 
    concatenate, 
    UpSampling2D, 
    Dropout,
    Lambda,
    Reshape,
    Flatten,
    Dense
)
from keras.layers.convolutional import (
    Conv2D, 
    MaxPooling2D,
)
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications import vgg16
from models.RoiPoolingConv import RoiPoolingConv

    
class EndToEndModel(object):

    def __init__(self, weights=None, gamma=3.0, pooling_regions=7, num_rois=1, theta=0.01, stage='train'):

        self.weights = weights
        self.stage = stage
        self.gamma = gamma
        self.pooling_regions = pooling_regions
        self.num_rois = num_rois
        self.theta = theta

    def binary(self, sample):
        return (sample * sample) / (sample * sample + self.theta * self.theta)

    def cal_salient_region(self, sample):

        sample = (sample - K.min(sample)) / (K.max(sample) - K.min(sample))

        mask = self.binary(sample)
        size = K.shape(sample)
        row = K.cast(tf.range(start=1, limit=size[0] + 1, delta=1), dtype='float32')
        row = K.reshape(row, (size[0], 1, 1))
        m01 = row * mask

        center_row = K.sum(m01) / K.sum(mask)
        val_row = (m01 - center_row) * mask
        val_row = K.sqrt(K.sum(val_row * val_row / K.sum(mask)))

        col = K.cast(tf.range(start=1, limit=size[1] + 1, delta=1), dtype='float32')
        col = K.reshape(col, (1, size[1], 1))
        m10 = col * mask
        center_col = K.sum(m10) / K.sum(mask)
        val_col = (m10 - center_col) * mask
        val_col = K.sqrt(K.sum(val_col * val_col / K.sum(mask)))
        h = K.cast(size[0], dtype='float32')
        w = K.cast(size[1], dtype='float32')
        start_row = K.minimum(K.maximum(0.0, center_row - val_row * self.gamma), h)
        start_col = K.minimum(K.maximum(0.0, center_col - val_col * self.gamma), w)
        height = K.minimum(K.maximum(0.0, val_row * self.gamma * 2.0), h - start_row)
        width = K.minimum(K.maximum(0.0, val_col * self.gamma * 2.0), w - start_col)

        sr = tf.convert_to_tensor([start_row, start_col, height, width])

        sr = tf.cast(sr, dtype='float32')
        sr = sr / 16.0
        return sr


    def cal_salient_regions(self, samples):
        salient_regions = tf.map_fn(lambda sample: self.cal_salient_region(sample), samples)
        return salient_regions

    def cal_salient_regions_output_shape(self, input_shape):
        return (input_shape[0], 1, 4)

    def EncodeLayer(self, input_tensor=None):
        input_shape = (None, None, 3)
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(
            img_input)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(
            conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(
            pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(
            conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1')(
            pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2')(
            conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block4_conv1')(
            pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block4_conv2')(
            conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='feature_map4')(conv4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block5_conv1')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block5_conv2')(conv5)

        return [conv1, conv2, conv3, conv4, conv5]

    def DecodeLayer(self, X):  # (conv1,conv2,conv3,drop4,drop5)

        up6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     name='block6_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_1')(X[4]))
        merge6 = concatenate([X[3], up6], axis=-1, name='concat_1')
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block6_conv2')(
            merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block6_conv3')(
            conv6)

        up7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     name='block7_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_2')(conv6))
        merge7 = concatenate([X[2], up7], axis=-1, name='concat_2')
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block7_conv2')(
            merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block7_conv3')(
            conv7)

        up8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     name='block8_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_3')(conv7))
        merge8 = concatenate([X[1], up8], axis=-1, name='concat_3')
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block8_conv2')(
            merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block8_conv3')(
            conv8)

        up9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block9_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_4')(conv8))
        merge9 = concatenate([X[0], up9], axis=-1, name='concat_4')
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block9_conv2')(
            merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block9_conv3')(
            conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block9_conv4')(
            conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', name='segmentation')(conv9)

        return conv10

    def AELayer(self, X, Y, stage='train'):
        sr = Lambda(self.cal_salient_regions, output_shape=self.cal_salient_regions_output_shape, name='saliency_box')(Y)

        out_roi_pool = RoiPoolingConv(self.pooling_regions, self.num_rois, name='roi_pooling')([X, sr])
        out = Flatten(name='flatten')(out_roi_pool)
        out = Dense(2048, activation='relu', name='fc1')(out)
        out = Dense(1024, activation='relu', name='fc2')(out)
        out = Dense(4, activation='linear', name='offset')(out)
        if stage == 'train':
            return out
        elif stage == 'test':
            return [out, sr]

    def BuildSaliencyModel(self):
        inputs = Input((None, None, 3))
        encoded_layer = self.EncodeLayer(inputs)
        decoded_layer = self.DecodeLayer(encoded_layer)
        model_saliency = Model(inputs, decoded_layer)
        if self.weights is not None:
            model_saliency.load_weights(self.weights)
        return model_saliency

    def BuildModel(self):

        model_saliency = self.BuildSaliencyModel()
        if self.weights is not None:
            model_saliency.load_weights(self.weights)
        saliency_input = model_saliency.get_layer('segmentation').output
        feature_input = model_saliency.get_layer('block5_conv2').output
        ae_layers = self.AELayer(feature_input, saliency_input, self.stage)
        model_total = Model(model_saliency.inputs, ae_layers)
        return model_total