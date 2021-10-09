# 对影像进行分块操作，将影像分成128*128的小影像，每相邻两张影像之间有16像元的重叠，这样的目的是为了方便拼接
import gdal
import osr
import os
import sys
import numpy as np


# 将整个大的遥感影像分解为128*128的小块，同时对样本数据进行分块，这样方便输入到模型中

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
    print("读取遥感影像信息结束")
    return im_proj, im_geotrans, im_width, im_height, im_data, dataset


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


# (717080.0, 10.0, 0.0, 2932970.0, 0.0, -10.0)
# GT2和GT4参数是0，而GT1是象元宽，GT5是象元高，（GT0，GT3）点位置是影像的左上角。
def update_geotrans(im_geotrans, width_index, height_index):
    a0 = im_geotrans[0] + im_geotrans[1] * width_index
    a3 = im_geotrans[3] + im_geotrans[5] * height_index
    update_geos = (a0, im_geotrans[1], im_geotrans[2], a3, im_geotrans[4], im_geotrans[5])
    return update_geos


if __name__ == '__main__':
    im_proj, im_geotrans, im_width, im_height, im_data, source_ds = read_img(
        r"D:\Work\line_extraction\ceshi1\source.tif")
    im_proj, im_geotrans, im_width, im_height, im_data, samples_ds = read_img(
        r"D:\Work\line_extraction\shp_to_raster\2.tif")
    # source_output_dir = r"D:\Work\line_extraction\segment_result\source"
    # sample_output_dir = r"D:\Work\line_extraction\segment_result\samples"
    source_output_dir = r"D:\Work\line_extraction\segment_result_train\source"
    sample_output_dir = r"D:\Work\line_extraction\segment_result_train\samples"
    # 记录当前的影像的索引值
    image_index = 0
    # 记录当前高度索引，从该索引出读取128像元高
    height_index = 0
    # 相邻小块的重叠像元个数
    stride = 96
    # 分割结果的边长像元
    length_of_side = 128
    everlay_pixel = length_of_side - stride
    while (height_index + length_of_side) <= im_height:
        # 记录当前宽度索引，从该索引出读取128像元宽
        width_index = 0
        while (width_index + length_of_side) <= im_width:
            source_data = source_ds.ReadAsArray(width_index, height_index, length_of_side, length_of_side)
            samples_data = samples_ds.ReadAsArray(width_index, height_index, length_of_side, length_of_side)
            b = np.sum(samples_data)
            if b != 0:
                tower_number = b / (25 * 255)
                if tower_number >= 5:
                    print("第" + str(image_index) + "个image对应的线塔大概是" + str(tower_number) + "个")
                    im_geotrans_update = update_geotrans(im_geotrans, width_index, height_index)
                    try:
                        # 保存文件的命名规则为image_index
                        writeimg(os.path.join(source_output_dir, str(image_index) + ".tif"), im_proj,
                                 im_geotrans_update,
                                 source_data)
                        writeimg(os.path.join(sample_output_dir, str(image_index) + ".tif"), im_proj,
                                 im_geotrans_update,
                                 samples_data)
                    except:
                        print("第" + str(image_index) + "出错")
                    image_index += 1
            # 112是128-16，减去步长
            width_index += everlay_pixel
        height_index += everlay_pixel

    print("分割结束，共" + str(image_index + 1) + "个小块")
