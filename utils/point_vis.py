import open3d as o3d
import numpy as np
import cv2

'''
input:1: a npy point cloud
      2: an rgb img
   
output: open3d point clouds with rgb color 
'''

def vis_point(img_path,point_cloud_path,output=None,isoutput =True,filter=100):

    # 读取图像和深度文件
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)[...,::-1]
    # 读取npy文件中的点云数据
    point_cloud_array = np.load(point_cloud_path)
    # 转换点云数组的形状
    if point_cloud_array.shape[-1]!=3:
        point_cloud_array = np.transpose(point_cloud_array)
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    # 将numpy数组中的数据复制到Open3D点云对象中
    pcd.points = o3d.utility.Vector3dVector(point_cloud_array)
    pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3) / 255.0)

    points = np.asarray(pcd.points)
    y_filter = points[:, 1] <= filter
    pcd = pcd.select_by_index(np.where(y_filter)[0])
    # 保存点云为PCD文件
    if isoutput:
        o3d.io.write_point_cloud(output, pcd)
        print("pcd saved in {}".format(output))
    return pcd

if __name__ == '__main__':

    imgpath = '/home/caojinghao/rail_tracking_and_obj_detection/temp/demo.jpg'
    point_cloud_array = "examplt_gt.npy"
    pcd = vis_point(imgpath,point_cloud_array,output=None,isoutput =False,filter=100)