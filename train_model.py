from model import unet, unet2, ceshiunet2, segnet_vgg16, fcn_vgg16_8s, VGGUnet2, D_resunet, D_resunet1, \
    fcn_2s  # ,res_unet
from data import trainGenerator, testGenerator, saveResult, testGenerator2
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

# 每轮训练的样本数量
batch_size = 1

# 训练数据集路径
train_path = r"D:\Work\line_extraction\segment_result_train\train"
valid_path = r'D:\Work\line_extraction\segment_result_train\valid'

data_gen_args = dict(rotation_range=90.,
                     # width_shift_range=0.1,
                     # height_shift_range=0.1,
                     # shear_range=0.1,
                     # zoom_range=0.1,
                     # fill_mode='nearest',
                     horizontal_flip=True,
                     vertical_flip=True)

# 训练数据集，第一个参数是batch-size
train_Gene = trainGenerator(2, train_path, 'img', 'mask', data_gen_args, save_to_dir=None)
# 验证数据集
val_Gene = trainGenerator(2, valid_path, 'img', 'mask', data_gen_args)

# 参考https://www.freesion.com/article/4104548785/
# monitor：监测的值，可以是accuracy，val_loss,val_accuracy
# factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
# patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
# mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率最小值，能缩小到的下限

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='min', epsilon=1e-4,
                              cooldown=0, min_lr=1e-6)

visual = TensorBoard(log_dir='./D_resunet1_log', histogram_freq=0, write_graph=True, write_images=True)
# 触发早停防止过拟合，monitor：需要监视的量，patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
# verbose信息展示模式， 0 为不在标准输出流输出日志信息，1 为输出进度条记录，2 为每个epoch输出一行记录

earlystop = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='min')


model = unet(input_size=(128, 128, 5))

# model.load_weights('res_unet.hdf5')
# ModelCheckpoint函数将在每个epoch后保存模型到filepath，monitor是需要监视的值，save_best_only为True时，将只保存在验证集上性能最好的模型
model_checkpoint = ModelCheckpoint(r'D:\Data\deepglobe-road-dataset\ceshi\model\mkdir\D_Resunet.hdf5',
                                   monitor='val_accuracy', verbose=1, mode='max', save_best_only=True)

print("开始训练数据")

# steps_per_epoch：当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
# epochs数据迭代的轮数
# validation_data：具有以下三种形式之一1.生成验证集的生成器2.个形如（inputs,targets）的tuple3.一个形如（inputs,targets，sample_weights）的tuple，这里是第一种
# validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数

model.fit_generator(train_Gene, steps_per_epoch=2, epochs=5,
                    #callbacks=[model_checkpoint, visual, reduce_lr, earlystop],
                    callbacks=[model_checkpoint],
                    validation_data=val_Gene,
                    validation_steps=1)

print("训练数据结束")
