import numpy as np
import random
import os
import sys

from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Dense
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import Image_Seg_Block
import tensorflow as tf

input_name = os.listdir(r'D:\Work\line_extraction\segment_result_train\train\img')
# 文件总的数目
n = len(input_name)
batch_size = 8
input_size_1 = 128
input_size_2 = 128
# 输入的波段数目
input_bands = 5
# 输出的波段数目
output_bands = 1
train_img = r"D:/Work/line_extraction/segment_result_train/train/img/"
train_mask = r"D:/Work/line_extraction/segment_result_train/train/mask/"
test_img = r"D:/Work/line_extraction/segment_result_train/valid/img/"
test_mask = r"D:/Work/line_extraction/segment_result_train/valid/mask/"


def batch_data(input_name, n, batch_size=8, input_size_1=128, input_size_2=128):
    rand_num = random.randint(0, n - 1)
    img1 = io.imread(train_img + input_name[rand_num]).astype("float")
    img2 = io.imread(train_mask + input_name[rand_num]).astype("float")
    # img1 = resize(img1, [input_size_1, input_size_2, 3])
    # img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, input_bands))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, output_bands))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0, n - 1)
        img1 = io.imread(train_img + input_name[rand_num]).astype("float")
        img2 = io.imread(train_mask + input_name[rand_num]).astype("float")
        # img1 = resize(img1, [input_size_1, input_size_2, 3])
        # img2 = resize(img2, [input_size_1, input_size_2, 3])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, input_bands))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, output_bands))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis=0)
        batch_output = np.concatenate((batch_output, img2), axis=0)
    return batch_input, batch_output


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


# '''
inpt = Input(shape=(input_size_1, input_size_2, input_bands))

conv1 = Conv2d_BN(inpt, 8, (3, 3))
conv1 = Conv2d_BN(conv1, 8, (3, 3))
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

conv2 = Conv2d_BN(pool1, 16, (3, 3))
conv2 = Conv2d_BN(conv2, 16, (3, 3))
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

conv3 = Conv2d_BN(pool2, 32, (3, 3))
conv3 = Conv2d_BN(conv3, 32, (3, 3))
pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

conv4 = Conv2d_BN(pool3, 64, (3, 3))
conv4 = Conv2d_BN(conv4, 64, (3, 3))
pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

conv5 = Conv2d_BN(pool4, 128, (3, 3))
conv5 = Dropout(0.5)(conv5)
conv5 = Conv2d_BN(conv5, 128, (3, 3))
conv5 = Dropout(0.5)(conv5)

convt1 = Conv2dT_BN(conv5, 64, (3, 3))
concat1 = concatenate([conv4, convt1], axis=3)
concat1 = Dropout(0.5)(concat1)
conv6 = Conv2d_BN(concat1, 64, (3, 3))
conv6 = Conv2d_BN(conv6, 64, (3, 3))

convt2 = Conv2dT_BN(conv6, 32, (3, 3))
concat2 = concatenate([conv3, convt2], axis=3)
concat2 = Dropout(0.5)(concat2)
conv7 = Conv2d_BN(concat2, 32, (3, 3))
conv7 = Conv2d_BN(conv7, 32, (3, 3))

convt3 = Conv2dT_BN(conv7, 16, (3, 3))
concat3 = concatenate([conv2, convt3], axis=3)
concat3 = Dropout(0.5)(concat3)
conv8 = Conv2d_BN(concat3, 16, (3, 3))
conv8 = Conv2d_BN(conv8, 16, (3, 3))

convt4 = Conv2dT_BN(conv8, 8, (3, 3))
concat4 = concatenate([conv1, convt4], axis=3)
concat4 = Dropout(0.5)(concat4)
conv9 = Conv2d_BN(concat4, 8, (3, 3))
conv9 = Conv2d_BN(conv9, 8, (3, 3))
conv9 = Dropout(0.5)(conv9)
# 由于我们要训练二分类模型，所以使用simoid函数，多分类模型则使用softmax函数
# filters是输出空间的维度
outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)
# outpt1=Dense(500,activation='relu')(outpt)
# outpt2=Dense(200,activation='relu')(outpt1)
# outpt3=Dense(2,activation='sigmoid')(outpt2)


model = Model(inpt, outpt)
model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])
# 通过model.summary()方法一览整个模型
model.summary()

itr = 50
S = []
loss_array=np.zeros(shape=(itr,),dtype=float)
for i in range(itr):
    print("iteration = ", i + 1)
    # 根据不同的训练次数设置不同的batch_size
    if i < 500:
        bs = 4
    elif i < 2000:
        bs = 8
    elif i < 5000:
        bs = 16
    else:
        bs = 32
    train_X, train_Y = batch_data(input_name, n, batch_size=bs)
    model.fit(train_X, train_Y, epochs=1, verbose=0)
    prediction = model.predict(train_X)
    # 计算损失值
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_Y, logits=prediction))
    loss_array[i]=loss
    print("iteration = "+str(i + 1) +"对应的loss值为："+str(loss))
    if i == (itr - 1):
        save_model(model, 'unet1.h5')
# '''
model = load_model('unet1.h5')


# sys.exit(5)

def batch_data_test(input_name, n, batch_size=8, input_size_1=128, input_size_2=128):
    rand_num = random.randint(0, n - 1)
    # 读取原始影像的值
    img1 = io.imread(test_img + input_name).astype("float")
    # 读取对应的样本数据的值
    img2 = io.imread(test_mask + input_name).astype("float")
    # 更改原始影像以及对应的样本数据的shape
    # img1 = resize(img1, [input_size_1, input_size_2, 3])
    # img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, input_bands))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, output_bands))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0, n - 1)
        img1 = io.imread(test_img + input_name[rand_num]).astype("float")
        img2 = io.imread(test_mask + input_name[rand_num]).astype("float")
        # img1 = resize(img1, [input_size_1, input_size_2, 3])
        # img2 = resize(img2, [input_size_1, input_size_2, 3])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, input_bands))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, output_bands))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis=0)
        batch_output = np.concatenate((batch_output, img2), axis=0)
    return batch_input, batch_output


test_name = os.listdir(test_img)
n_test = len(test_name)
sample_output_dir = r"D:\Work\line_extraction\segment_result_train\pred"
for i in range(n_test):
    im_proj, im_geotrans, im_width, im_height, im_data, source_ds = Image_Seg_Block.read_img(
        os.path.join(test_img, test_name[i]))
    test_X, test_Y = batch_data_test(test_name[i], 1, batch_size=1)
    # 进行预测
    pred_Y = model.predict(test_X)
    print("预测结束")
    pred_Y_shape = np.array(pred_Y).reshape(im_height, im_width)
    Image_Seg_Block.writeimg(os.path.join(sample_output_dir, test_name[i]), im_proj,
                             im_geotrans, pred_Y_shape)

ii = 0
plt.figure()
# plt.imshow(test_X[ii, :, :, :])
# plt.axis('off')
# plt.figure()
# plt.imshow(test_Y[ii, :, :, :])
# plt.axis('off')
# plt.figure()
# plt.imshow(pred_Y[ii, :, :, :])
# plt.axis('off')
# plt.show()
