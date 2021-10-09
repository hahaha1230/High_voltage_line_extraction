# 如何使用GDAL/OGR打开矢量文件、读取属性表字段，并将数据范围和每个ploygon要素的范围
import ogr, sys, os
import numpy as np
import gdal
import osr
# 写入shp文件,polygon
def writeShp():
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 属性表字段支持中文
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    # 注册驱动
    ogr.RegisterAll()
    # 创建shp数据
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        return "驱动不可用："+strDriverName
    # 创建数据源
    oDS = oDriver.CreateDataSource(r"D:\Work\高压线提取\samples\all\polygon.shp")
    if oDS == None:
        return "创建文件失败：polygon.shp"
    # 创建一个多边形图层，指定坐标系为WGS84
    papszLCO = []
    geosrs = osr.SpatialReference()
    #geosrs.SetWellKnownGeogCS("WGS84")
    # 线：ogr_type = ogr.wkbLineString
    # 点：ogr_type = ogr.wkbPoint
    ogr_type = ogr.wkbPolygon
    # 面的类型为Polygon，线的类型为Polyline，点的类型为Point
    oLayer = oDS.CreateLayer("Polygon", geosrs, ogr_type, papszLCO)
    if oLayer == None:
        return "图层创建失败！"
    # 创建属性表
    # 创建id字段
    oId = ogr.FieldDefn("id", ogr.OFTInteger)
    oLayer.CreateField(oId, 1)
    # 创建name字段
    oName = ogr.FieldDefn("name", ogr.OFTString)
    oLayer.CreateField(oName, 1)
    oDefn = oLayer.GetLayerDefn()
    # 创建要素
    # 数据集
    # wkt_geom id name
    features = ['test0;POLYGON((700890 2946910, 700890 2855640,774940.0 2855640.0,774940.0 2946910.0,700890 2946910))']
    for index, f in enumerate(features):
        oFeaturePolygon = ogr.Feature(oDefn)
        oFeaturePolygon.SetField("id",index)
        oFeaturePolygon.SetField("name",f.split(";")[0])
        geomPolygon = ogr.CreateGeometryFromWkt(f.split(";")[1])
        oFeaturePolygon.SetGeometry(geomPolygon)
        oLayer.CreateFeature(oFeaturePolygon)
    # 创建完成后，关闭进程
    oDS.Destroy()
    return "数据集创建完成！"
writeShp()

os.chdir(r'D:\Work\高压线提取\samples\all')

# 设置driver,并打开矢量文件
driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open('5.shp', 0)
if ds is None:
    print("Could not open", 'sites.shp')
    sys.exit(1)
# 获取图册
layer = ds.GetLayer()

# 要素数量
numFeatures = layer.GetFeatureCount()
print("Feature count: " + str(numFeatures))

# 获取范围
extent = layer.GetExtent()
print("Extent:", extent)
print("UL:", extent[0], extent[3])
print("LR:", extent[1], extent[2])


# 获取要素
# feature = layer.GetNextFeature()

# 循环每个要素属性
# for i in range(numFeatures):
#     feature = layer.GetNextFeature()
#     # 获取字段“id”的属性
#     id = feature.GetField('type')
#     # 获取空间属性
#     print(id)
#     geometry = feature.GetGeometryRef()
#     # x = geometry.GetX()
#     polygonextent = geometry.GetEnvelope()
#     print(geometry.GetEnvelope())
#     # print(y)
#
#     # y = geometry.GetY()
#     print("UL:", polygonextent[0], polygonextent[3])
#     print("LR:", polygonextent[1], polygonextent[2])

