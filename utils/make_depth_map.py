import cv2
import numpy as np
from scipy.spatial.transform import Rotation

'''
#NOTE
this script make a depth map of an input image 
according to the instrinsic and extrinsic parameters
of the input image
author Jinghao Cao
'''

def make_depth_map(para_dict,mask,output=None,isoutput=False,isline=True,debug=False):
   #NOTE make a rail view in a virtual plane
    hfov = para_dict['hfov']
    vfov = para_dict['vfov']
    camera_height = para_dict['camera_height']

    h,w = para_dict["image_shape"]

    cx = h/2
    cy = w/2

    fx = (h/2)/np.tan(np.deg2rad(hfov/2))
    fy = (w/2)/np.tan(np.deg2rad(vfov/2))
    K_src = [[fx,0,cx],[0,fy,cy],[0,0,1]]

    x,y,z = para_dict['euler']

    r =Rotation.from_euler('zyx',[z,y,x],degrees=True)
    R = r.as_matrix()  #欧拉角得到旋转矩阵
    t = para_dict["tr"]

    imgshape = (h,w)
    #rail_img = cv2.imread("/home/caojinghao/rail_tracking_and_obj_detection/railway_view.png")
    deg = abs(x) # 旋转角的标量
    depth_z = []
    for j in range(h):
      
      depth_z.append(camera_height/(np.tan(np.deg2rad(deg+(hfov*(j/(h-1)-1/2))))))

    depth_z = np.array(depth_z)
    depth_z[depth_z<=0]=np.inf  
    depth_z = depth_z[:,np.newaxis]

    depth_z = np.tile(depth_z,(1,w))
    
    distance = np.sqrt(np.multiply(depth_z,depth_z)+np.ones_like(depth_z)*camera_height**2)
    theta = -((np.arange(w)/(w-1)-0.5)*hfov)
    theta = theta[np.newaxis,:]
    theta = np.tile(theta,(h,1))

    depth_x = np.multiply(distance,np.tan(np.deg2rad((theta))))

    if not isline:
       mask[mask!=255]=0
    if len(mask.shape)==3:
      mask=mask[:,:,0]
    mask=(mask).astype(np.bool_)

    depth_z[mask]=np.inf

    if debug:
       mask = np.logical_and(abs(depth_x<3),abs(depth_z<100)).astype(np.int8)
       cv2.imwrite("./railmask.jpg",mask*255)


    X = depth_x.reshape(1,-1)
    Z = depth_z.reshape(1,-1)
    Y = np.zeros_like(X)

    world_point = np.concatenate((X,Y,Z)).astype(np.float32)
    if isoutput:  
      np.save(output,world_point)
    
    return world_point.T

if __name__ == '__main__':
  dict = {}
  dict['hfov'] = 105
  dict['vfov'] = 73
  dict['camera_height'] =1.5
  dict['image_shape']=(1080,1920)

  dict['euler']=(-4.5,0,0)
  dict['tr'] =np.array([0,1.5,0]).reshape(3,1) #先转后移动
  img ="/home/caojinghao/railway_object_detection/demo_image/rail_mask.jpg"
  mask = cv2.imread(img)
  output = "./example_gt.npy"
  world_point = make_depth_map(dict,mask,output,isline=True)

  
    