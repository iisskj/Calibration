import open3d as o3d
import copy
import numpy as np

'''
#TODO 这个程序是在模仿相机和雷达的联合标定,目前为手动调整,后续改善
'''
def rotate(pcd_left, pcd_right,T):
    '''
    T=[R,tr
       0,1]
    '''
    origin_point = (0,0,0)
    ## left rail
    pcd_transform_left= copy.deepcopy(pcd_left)
    pcd_transform_left.scale(1.535, center=origin_point)
    pcd_transform_left = pcd_transform_left.transform(T)
    # pcd_combine = pcd_transform + rail
    pcd_combine_left = pcd_transform_left

    ## right_rail
    pcd_transform_right = copy.deepcopy(pcd_right)
    pcd_transform_right.scale(1.535, center=origin_point)
    pcd_transform_right = pcd_transform_right.transform(T)
    # pcd_combine = pcd_transform + rail
    pcd_combine_right = pcd_transform_right

    return pcd_combine_left,pcd_combine_right

if __name__ == '__main__':
    pcd_left = o3d.io.read_point_cloud("./data/point/point_cloud_crooked_left.pcd")
    pcd_right = o3d.io.read_point_cloud("./data/point/point_cloud_crooked_right.pcd")
    T = np.array([[-0.997564077377, -0.069756045938, 0.000243332892, 0.750000000000],
                    [-0.000000150628, 0.003490474075, 0.999993920326, -0.879999995232],
                    [-0.069756470621, 0.997557997704, -0.003481982043, 20.000000000000],
                    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])
    left,right = rotate(pcd_left, pcd_right,T)