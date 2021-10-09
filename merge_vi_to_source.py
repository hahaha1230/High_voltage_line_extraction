import numpy as np
import gdal
# 读取遥感影像
def read_img(filename):
    # gdal打开影像数据
    dataset = gdal.Open(filename)
    # 获取影像宽度
    im_width = dataset.RasterXSize
    # 获取影像高度
    im_height = dataset.RasterYSize
    # 获取影像地理参考
    im_geotrans = dataset.GetGeoTransform()
    # 获取影像投影坐标系
    im_proj = dataset.GetProjection()
    # 获取影像数据
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    # 释放掉dataset资源
    del dataset
    return im_proj, im_geotrans, im_data, im_height, im_width


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

if __name__ == '__main__':
    proj, geotrans, ym_shadow, height, width = read_img(
        r'D:\RF_castcorrect\SourceImage\band\shadow.tif')