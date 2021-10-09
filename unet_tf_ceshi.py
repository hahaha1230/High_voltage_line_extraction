import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np

# import tensorflow.compat.v1 as tf
#
# tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.utils import np_utils
import datetime, gdal, os
import random
from skimage import io
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Dense
import pandas as pd


# 读取对应位置的遥感影像像元值
def read_img(filename):
    # print("正在读取遥感影像信息")
    # 利用gdal打开遥感影像数据
    dataset = gdal.Open(filename)
    # 获取影像的宽度
    im_width = dataset.RasterXSize
    # 获取影像的高度
    im_height = dataset.RasterYSize
    # 获取影像的地理参考
    im_geotrans = dataset.GetGeoTransform()
    # 获取影像的坐标系
    im_proj = dataset.GetProjection()
    # 将遥感影像的像元值读取到数组里面
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # 删除变量
    del dataset
    print("读取遥感影像信息结束")
    return im_proj, im_geotrans, im_width, im_height, im_data


# 将内存中的影像数据写入到文件中
def writeimg(filename, im_proj, im_geotrans, im_data):
    # 确定影像的保存类型，预测结果大部分是浮点型，所以类型大概率也是浮点型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 获取gdal的tif格式驱动
    driver = gdal.GetDriverByName("GTiff")
    # 创建对应的文件
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    # 为新创建的tif文件设置坐标系和投影坐标系
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    # 判断是否为单波段，如果不是则将数据分别写入到对应的波段中
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    # 释放掉资源
    del dataset


# 指定要分类的类别个数
class_num = 1
input_dimension = 3
output_dimension = 3
# 定义每个批次大小
batch_size = 20
# 训练次数
max_step = 300
# 文件路径
DIR = "D:\Work\shimohua\model"

input_size_1 = 1024
input_size_2 = 1024

# 定义一个命名空间
with tf.name_scope("input"):
    # 定义两个占位变量
    input_x = tf.placeholder(tf.float32, [None, input_size_1, input_size_2, input_dimension], name="x-input")
    input_y = tf.placeholder(tf.float32, [None, input_size_1, input_size_2, output_dimension], name="y-input")
# 设置参数设置DROPOUT参数
arg_dropout = tf.placeholder(tf.float32)

# 定义神经网络
with tf.name_scope("layer"):
    # def Define_model():
    ## unet网络结构下采样部分
    # 输入层 第一部分
    # inputs = tf.keras.layers.Input(shape=(1024, 1024, 3))
    # x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(input_x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 256*256*64
    # 下采样
    x1 = tf.keras.layers.MaxPooling2D(padding="same")(x)  # 128*128*64

    # 卷积 第二部分
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128
    # 下采样
    x2 = tf.keras.layers.MaxPooling2D(padding="same")(x1)  # 64*64*128

    # 卷积 第三部分
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256
    # 下采样
    x3 = tf.keras.layers.MaxPooling2D(padding="same")(x2)  # 32*32*256

    # 卷积 第四部分
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512
    # 下采样
    x4 = tf.keras.layers.MaxPooling2D(padding="same")(x3)  # 16*16*512
    # 卷积  第五部分
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024

    ## unet网络结构上采样部分

    # 反卷积 第一部分      512个卷积核 卷积核大小2*2 跨度2 填充方式same 激活relu
    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x4)  # 32*32*512
    x5 = tf.keras.layers.BatchNormalization()(x5)
    x6 = tf.concat([x3, x5], axis=-1)  # 合并 32*32*1024
    # 卷积
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)  # 32*32*512

    # 反卷积 第二部分
    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x6)  # 64*64*256
    x7 = tf.keras.layers.BatchNormalization()(x7)
    x8 = tf.concat([x2, x7], axis=-1)  # 合并 64*64*512
    # 卷积
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)  # #64*64*256

    # 反卷积 第三部分
    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x8)  # 128*128*128
    x9 = tf.keras.layers.BatchNormalization()(x9)
    x10 = tf.concat([x1, x9], axis=-1)  # 合并 128*128*256
    # 卷积
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)  # 128*128*128

    # 反卷积 第四部分
    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,
                                          padding="same",
                                          activation="relu")(x10)  # 256*256*64
    x11 = tf.keras.layers.BatchNormalization()(x11)
    x12 = tf.concat([x, x11], axis=-1)  # 合并 256*256*128
    # 卷积
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)  # 256*256*64

    # 输出层 第五部分
    output = tf.keras.layers.Conv2D(3, 1, padding="same", activation="softmax")(x12)  # 256*256*34

    # return tf.keras.Model(inputs=inputs, outputs=output)

with tf.name_scope("loss"):
    # 方法一：二次代价函数,平方差作为loss值
    # loss = tf.reduce_mean(tf.square(prediction - y))
    # 方法二：交叉墒,使用softmax回归之后的交叉熵损失函数
    # print("计算损失函数：labels"+input_y.shape+"logits:"+output.shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=output))
    tf.summary.scalar("loss", loss)

# 加载模型直接用于预测
# def load_model_pred():
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(r"D:\Work\shimohua\model\projector")
#         proj, geotrans, width, height, data = read_img(input_tif_location)
#         # data = np.array([[502, 570, 583, 1629]])
#         print(ckpt)
#         if ckpt and ckpt.all_model_checkpoint_paths:
#             path = ckpt.all_model_checkpoint_paths[0]
#             saver.restore(sess, path)
#             global_step = path.split('/')[-1].split('-')[-1]
#             pred_result = sess.run([prediction], feed_dict={x: data, arg_dropout: 1})
#             return pred_result
#             # 加载模型
#             # 这一部分是有多个模型文件时，对所有模型进行测试验证
#             # for path in ckpt.all_model_checkpoint_paths:
#             #     saver.restore(sess, path)
#             #     global_step = path.split('/')[-1].split('-')[-1]
#             #     test_pre = sess.run([prediction], feed_dict={x: data, arg_dropout: 1})
#         else:
#             print('没有找到模型的权重文件')
#             return None


# adam梯度下降方式最小化代价函数
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
# 合并所有的summary标量，
merged = tf.summary.merge_all()


def get_batch_data(train_input_names, val_input_names, batch_size, now_batch, total_batch):
    img1 = io.imread(train_img + train_input_names[0]).astype("float")
    img2 = io.imread(train_mask + val_input_names[0]).astype("float")
    # img1 = resize(img1, [input_size_1, input_size_2, 3])
    # img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, input_bands))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, output_bands))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        img1 = io.imread(train_img + train_input_names[now_batch]).astype("float")
        img2 = io.imread(train_mask + val_input_names[now_batch]).astype("float")
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, input_bands))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, output_bands))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis=0)
        batch_output = np.concatenate((batch_output, img2), axis=0)
        now_batch += 1
    return batch_input, batch_output, now_batch


# 按照正确分类的像元个数计算正确精度
def calc_accu(sample, pred):
    accu_pixel_num = 0
    for i in range(1024):
        for j in range(1024):
            if sample[i, j] == pred[i, j]:
                accu_pixel_num += 1
    return accu_pixel_num / (1024 * 1024)


def train_model():
    n_batch = n // batch_size
    with tf.Session() as sess:
        # f.global_variables_initializer()初始化模型的参数， 整句话就是run了 所有global Variable 的 assign op
        sess.run(tf.global_variables_initializer())
        # 将对应的sess.graph文件写入到logs文件夹下面
        write = tf.summary.FileWriter("logs/", sess.graph)
        # 训练精度,精度的话采用正确预测出来的像元个数占用总的应该预测像元个数的比例来代替
        train_accu = np.zeros([max_step], dtype=float)
        # 测试精度，测试精度的话采用
        test_accu = np.zeros([max_step], dtype=float)
        # 开始训练时间
        train_start_time = datetime.datetime.now()
        # 创建一个数组记录每次的训练损失率
        train_loss_list = []
        # 创建一个数组记录每次epoch中所有的train_loss均值
        train_loss_epoch = []

        # 对于训练数据执行max_step训练过程
        for epoch in range(max_step):
            now_batch = 0
            # 对于每次的训练过程
            for batch in range(n_batch):
                # 从now_batch索引开始获取batch_size个数据进行训练
                # 输入的shape应该是 (batch, rows, cols, channels)
                batch_xs, batch_ys, now_batch = get_batch_data(train_input_names, train_mask_input_names, batch_size,
                                                               now_batch, n_batch)
                # 执行模型的训练和预测
                summary, _, train_loss, pred = sess.run([merged, train, loss, output],
                                                        feed_dict={input_x: batch_xs, input_y: batch_ys,
                                                                   arg_dropout: 1})
                # _, train_loss, pred = sess.run([train, loss, output],
                #                                         feed_dict={x: batch_xs, y: batch_ys, arg_dropout: 1})
                print("当前batch：" + str(now_batch) + "对应的训练的损失值为" + train_loss)
                # 将计算出的训练损失率添加到数组中
                train_loss_list.append(train_loss)
                accu_pixel = calc_accu(batch_ys, pred)
                print("第" + str(epoch) + "epoch结束，对应的的正确分类的像元个数所占百分比是" + str(accu_pixel))
                # print("正确分类的像元个数占据总的正确像元个数的比例是"+str(accu_pixel))
                # 计算训练的精度情况，用正确分类像元所占比例来作为精度
                train_accu[epoch] = accu_pixel

            # 记录每次epoch中所有的train_loss均值
            train_loss_epoch.append(np.mean(train_loss_list))
            if epoch % 10 == 0:
                print('epoch  ' + str(epoch) + ' train_loss ' + str(np.mean(train_loss_list)))

            # print("epoch=" + str(epoch) + "训练数据的正确率" + str(train_true_number / X_train.shape[0]))
            print("开始计算验证数据集的精度")
            now_val_batch = 0
            for val_batch in range(len(val_input_names) // batch_size):
                # 逐个遍历获取测试数据集的每个batch
                batch_xss, batch_yss, now_val_batch = get_batch_data(val_input_names, val_mask_names, batch_size,
                                                                     now_val_batch, n_batch)
                # 在使用 Session 对象的 run()调用执行图时，传入 tensor帮助取回结果
                # 这里的feed是使用一个值临时替换一个另一个值，比如这里就是用batch_xss代替x.  feed 只在调用它的方法内有效, 方法结束, feed 就会消失.
                test_pre, test_loss, pred = sess.run([pred, loss, output],
                                                     feed_dict={input_x: batch_xss, input_y: batch_yss, arg_dropout: 1})
                # 计算测试的精度情况，用正确分类像元所占比例来作为精度
                test_accu[epoch] = calc_accu(batch_yss, pred)
            # print("epoch="+str(epoch)+"测试集上面精度为" + str(test_true_number / X_test.shape[0]))
        plt.plot(range(max_step), train_loss_epoch, color='r')
        plt.xlabel('epoch')
        plt.ylabel('train_loss_mean')
        plt.show()
        # 结束训练时间
        train_end_time = datetime.datetime.now()
        print("模型训练总用时：" + str(train_end_time - train_start_time) + "s")
        print("训练集上最好的精度是:" + str(train_accu.max()) + "对应的epoch是：" + str(train_accu.argmax()))
        print("测试集上最好的精度是:" + str(test_accu.max()) + "对应的epoch是：" + str(test_accu.argmax()))
        plt.plot(range(max_step), train_accu, color='r')
        plt.plot(range(max_step), test_accu, color='b')

        plt.xlabel('epoch')
        plt.ylabel('accu')
        # plt.title('精度随着epoch的变化')
        plt.show()
        plt.savefig(r"D:\Work\shimohua\Figure_6.png")
        # plt.savefig("Figure_6.png")
        saver.save(sess, DIR + "/projector/a_model_ckpt", global_step=max_step)

        # print("对整个影像预测开始")
        # 记录模型预测开始时间
        # pred_start_time = datetime.datetime.now()
        # # 记录最终的预测结果
        # result = np.zeros([height, width], dtype=int)
        #
        # for h in range(height):
        #     # print("当前进度：" + str(h / height))
        #     input = norm_image_data_vi[:, h:h + 1, :]
        #     # 将数据转换为按照像元排列格式，转换之前是按照波段排列
        #     input_trans = input.transpose(2, 1, 0)
        #     input_trans1 = input_trans.reshape(-1, input_dimension)
        #
        #     # 对整个影像数据执行预测操作，由于直接输入整个影像会导致内存不足，这里
        #     test_pre = np.array(
        #         sess.run([prediction], feed_dict={x: input_trans1, arg_dropout: 1}),
        #         dtype=float)
        #     # 更改预测结果的shape以便我们可以获取预测的最终结果，这里是从（1，X_train.shape[0],class_num)变为（X_train.shape[0],class_num)
        #     test_pre1 = test_pre.reshape(-1, class_num)
        #     # test_pre1里面记录每个像元归属于不同类别的概率，获取预测结果的概率最大值对应的值也即对应的类别-1
        #     result[h:h + 1, :] = np.array(np.argmax(test_pre1, axis=1) + 1).reshape(1, -1)
        #
        # # 记录预测结束时间
        # pred_end_time = datetime.datetime.now()
        # print("预测总用时：" + str(pred_end_time - pred_start_time) + 's')
        # write.close()
        # return result


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # train_input_names = os.listdir(r'D:\Work\line_extraction\segment_result_train\train\img')
    # val_input_names = os.listdir(r"D:/Work/line_extraction/segment_result_train/valid/img")
    train_input_names = os.listdir(r'D:/Data/deepglobe-road-dataset/ceshi/sourceimage/train/img')
    train_mask_input_names = os.listdir(r'D:/Data/deepglobe-road-dataset/ceshi/sourceimage/train/mask')
    val_input_names = os.listdir(r"D:/Data/deepglobe-road-dataset/ceshi/sourceimage/valid/img")
    val_mask_names = os.listdir(r"D:/Data/deepglobe-road-dataset/ceshi/sourceimage/valid/mask")
    # 文件总的数目
    n = len(train_input_names)
    batch_size = 2
    # input_size_1 = 128
    # input_size_2 = 128
    # # 输入的波段数目
    # input_bands = 5
    input_size_1 = 1024
    input_size_2 = 1024
    # 输入的波段数目
    input_bands = 3
    # 输出的波段数目
    output_bands = 3
    # train_img = r"D:/Work/line_extraction/segment_result_train/train/img/"
    # train_mask = r"D:/Work/line_extraction/segment_result_train/train/mask/"
    # test_img = r"D:/Work/line_extraction/segment_result_train/valid/img/"
    # test_mask = r"D:/Work/line_extraction/segment_result_train/valid/mask/"

    train_img = r"D:/Data/deepglobe-road-dataset/ceshi/sourceimage/train/img/"
    train_mask = r"D:/Data/deepglobe-road-dataset/ceshi/sourceimage/train/mask/"
    test_img = r"D:/Data/deepglobe-road-dataset/ceshi/sourceimage/valid/img/"
    test_mask = r"D:/Data/deepglobe-road-dataset/ceshi/sourceimage/valid/mask/"
    # Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(r"D:\Data\deepglobe-road-dataset\ceshi\sourceimage\model")
    # 如果在对应的文件夹下面获取的文件不为空，就去加载该模型并直接进行预测
    # if ckpt == None:
    #     sess=train_model()
    #     pred_result=predict_data(sess,data)
    # else:
    #     pred_result=load_model_pred(data)
    pred_result = train_model()

    # writeimg(output_location, proj, geotrans, pred_result.reshape(height, width))

    # 输出测试结果数据集的损失均值
    # print('test_loss  ' + str(np.mean(test_loss_list)))

    # plt.plot(range(X_test.shape[0]), true, 'b-')
    # plt.plot(range(X_test.shape[0]), pre, 'r:')
    # #将图片保存
    # plt.savefig('./shimohua/test2.jpg')
    # #展示效果图片
    # plt.show()
    # 保存server

    endtime = datetime.datetime.now()
    print("总用时：" + str(endtime - starttime) + 's')
