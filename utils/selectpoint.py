import open3d as o3d
import numpy as np

def selectpoint(input_points,xlim=None,ylim=None,zlim=None):

    if xlim:
        mask = abs(input_points[:,0])<xlim
        #mask = np.repeat(mask,3,axis=1)
        input_points = input_points[mask]
    if ylim:
        mask = abs(input_points[:,1])<ylim
        #mask = np.repeat(mask,3,axis=1)
        input_points = input_points[mask]
    if zlim:
        mask = abs(input_points[:,2])<zlim
        #mask = np.repeat(mask,3,axis=1)
        input_points = input_points[mask]
    
    return input_points