#! python
# -*- coding: utf-8 -*-

"""
@Py-V  : 3
@File  : gif2mnist.py
@Author: lanliusong
@Date  : 2019/10/22 14:06
@Ide   : PyCharm
@Desc  : 描述...
"""

import glob
import numpy as np
from PIL import Image

if __name__ == '__main__':
    path = 'img/*.gif'
    x_train = []
    n = len(glob.glob(path))
    for index, imageFile in enumerate(glob.glob(path)):
        # 打开图像并转化为数字矩阵
        img = np.array(Image.open(imageFile))
        print('%d/%d' % (index + 1, n))
        if len(x_train):
            x_train = np.vstack((x_train, img))
        else:
            x_train = img

    print(x_train.shape)

    np.save('./x_train.py.bak', x_train)
