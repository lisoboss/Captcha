#! python
# -*- coding: utf-8 -*-

"""
@Py-V  : 3.7
@File  : train.py
@Author: lanliusong
@Date  : 2019/10/22 10:48
@Ide   : PyCharm
@Desc  : 训练模型描述...
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow_core.python.keras import layers, models, optimizers


Height = 60
Width = 200
Channels = 1
Train_batch = 6000
Test_batch = 1000


class CNN(object):
    def __init__(self):
        model = models.Sequential(name='ICP_model')
        # 第1层卷积，卷积核大小为3*3，32个，60*200为待训练图片的大小
        model.add(layers.Conv2D(32, (3, 3), activation='re' + 'lu', input_shape=(Height, Width, Channels)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(32, (3, 3), activation='re' + 'lu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='re' + 'lu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第4层卷积，卷积核大小为3*3，128个
        model.add(layers.Conv2D(128, (3, 3), activation='re' + 'lu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(6 * 62, activation='re' + 'lu'))
        model.add(layers.Reshape([6, 62]))

        model.add(layers.Softmax())

        # 打印网络的字符串摘要
        model.summary()

        self.model = model


class DataSource(object):
    def __init__(self):

        # 加载数据
        data_path = './img/'
        data_img = [name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))]

        data_img = np.array(data_img)

        np.random.shuffle(data_img)

        # 6千张训练图片，1千张测试图片

        train_images = []
        train_labels = []

        for _i, fn in enumerate(data_img):
            if _i == Train_batch:
                break
            train_images.append(np.array(Image.open(os.path.join(data_path, fn)).convert('L'), np.float))
            train_labels.append(self.name2array(fn.split('.')[0]))

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        np.random.shuffle(data_img)

        test_images = []
        test_labels = []

        for _i, fn in enumerate(data_img):
            if _i == Test_batch:
                break
            test_images.append(np.array(Image.open(os.path.join(data_path, fn)).convert('L'), np.float))
            test_labels.append(self.name2array(fn.split('.')[0]))

        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        train_images = train_images.reshape((Train_batch, Height, Width, Channels))
        test_images = test_images.reshape((Test_batch, Height, Width, Channels))

        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

    @staticmethod
    def name2array(name=None):

        a = []
        for _n in name:
            if ord('0') <= ord(_n) <= ord('9'):
                a.append(ord(_n) - ord('0'))
            elif ord('A') <= ord(_n) <= ord('Z'):
                a.append(ord('9') - ord('0') + ord(_n) - ord('A') + 1)
            else:
                a.append(ord('9') - ord('0') + ord('Z') - ord('A') + ord(_n) - ord('a') + 2)

        return np.array(a)


class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        n = 100

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                           save_weights_only=True,
                                                           verbose=1,
                                                           save_freq=Train_batch * 10)

        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

        self.cnn.model.fit(self.data.train_images,
                           self.data.train_labels,
                           shuffle=True,
                           epochs=n,
                           callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == '__main__':
    # CNN()
    app = Train()
    app.train()
