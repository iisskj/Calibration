
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

'''
This script is used to fit the track line curve
'''
def polyfit_railway(points, degree, testmode=False,ispcd =True,isright=False):
    
    # 获取点云数据
    if ispcd:
        points = np.asarray(points.points)

    # 提取x和z坐标
    x = points[:, 0]
    z = points[:, 2]

    # 将点转换成二维点
    points_2d = np.column_stack((x, z))

    # 对每组数据分别拟合一条曲线
    #coeffs1 = np.polyfit(points_2d[:, 0], points_2d[:, 1], degree)
    coeffs1 = np.polyfit(points_2d[:, 1], points_2d[:, 0], degree)
    
    if testmode:
        # 生成拟合的点
        z_fit1 = np.linspace(np.min(points_2d[:, 1]), np.max(points_2d[:, 1]), 1000)
        x_fit1 = np.polyval(coeffs1, z_fit1)
        # 绘制原始点和拟合的曲线
        plt.scatter(x, z, s=0.6)
        plt.plot(x_fit1, z_fit1, 'r')
        plt.xlim(-3, 3)
        plt.gca().invert_xaxis()

            #debugger
        ori_pcd = o3d.geometry.PointCloud()
        ori_pcd.points = o3d.utility.Vector3dVector(points)
        
        if isright :
            o3d.io.write_point_cloud("/home/caojinghao/railway_object_detection/data/output/right_ori.pcd",ori_pcd)
            plt.savefig("./right_poly.jpg")
        else:
            o3d.io.write_point_cloud("/home/caojinghao/railway_object_detection/data/output/left_ori.pcd", ori_pcd)
            plt.savefig("./left_poly.jpg")
    
    Z = np.linspace(np.min(points_2d[:, 1]), np.max(points_2d[:, 1]), 1000)
    X = np.polyval(coeffs1,Z)
    Y = np.zeros_like(X)
    
    line_points = np.stack((X,Y,Z),axis=-1)
    # 将数据存储为Open3D点云对象
    railline_pcd = o3d.geometry.PointCloud()
    railline_pcd.points = o3d.utility.Vector3dVector(line_points)

    return railline_pcd


if __name__ == '__main__':
# 读取pcd点云文件
    pcd = o3d.io.read_point_cloud("./data/point/pcd_crooked_left.pcd")
    degree = 2
    rail_line = polyfit_railway(pcd, degree, testmode=False)