import numpy as np
import open3d as o3d
import os
import sys

sys.path.append(os.path.abspath('/home/caojinghao/railway_object_detection/'))
from utils.make_depth_map import *
from utils.point_vis import *
from geography.segment_lr import *
from geography.polyfit import *
from utils.selectpoint import *

#TODO 以后这里的mask改成用图像直接分割出来
mask = cv2.imread("/home/caojinghao/railway_object_detection/data/input/mask1.png",cv2.IMREAD_GRAYSCALE)
if mask.shape!=(1080,1920):
    mask=cv2.resize(mask,(1920,1080),interpolation=cv2.INTER_NEAREST)
    print("warning! image resized!!!")

left_rail,right_rail = segment_lr(mask,1400,False)

dict = {}
dict['hfov'] = 105
dict['vfov'] = 73
dict['camera_height'] =1.5
dict['image_shape']=(1080,1920)

dict['euler']=(-4.5,0,0)
dict['tr'] =np.array([0,1.5,0]).reshape(3,1) #先转后移动
img ="/home/caojinghao/railway_object_detection/demo_image/rail_mask.jpg"
output = "./example_gt.npy"


left_rail = make_depth_map(dict,left_rail)
left_rail = selectpoint(left_rail,None,None,200)

railline_pcd = polyfit_railway(left_rail,degree=2,ispcd=False,testmode=False,isright=False)
o3d.io.write_point_cloud("/home/caojinghao/railway_object_detection/data/output/left.pcd", railline_pcd)


right_rail = make_depth_map(dict,right_rail)
right_rail = selectpoint(right_rail,None,None,200)
railline_pcd = polyfit_railway(right_rail,degree=2,ispcd=False,testmode=False,isright=True)
o3d.io.write_point_cloud("/home/caojinghao/railway_object_detection/data/output/right.pcd", railline_pcd)

print("ok")
