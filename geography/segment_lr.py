# 导入所需的库
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
input :
1: segmented mask
2: roi limitation

output:
left rail line img
right rail ine img
'''

def segment_lr(img ,roi_limit,ismakescatter=False):

    # 读取图片并转换为灰度图

    height, width = img.shape
    ## np.unique从上到下从左到右读取数据，左右翻转前得到的是左边铁轨，翻转后得到右边铁轨
    img_flip = cv2.flip(img,1)
    # 将图片转换为数组
    img_array = np.array(img)
    img_flip_array = np.array(img_flip)

    # 找到灰度值为0的点的坐标
    y, x = np.where(img_array == 0)
    y_flip, x_flip = np.where(img_flip_array == 0)

    # 找到每一行中灰度值为0的点的x值最小的点的坐标
    unique_y, index = np.unique(y, return_index=True)
    unique_y_flip, index_flip = np.unique(y, return_index=True)
    # unique_y, index = np.flip((y, return_index=True))


    unique_x = x[index]
    unique_x_flip = x_flip[index_flip]

    mask = abs(unique_x)<roi_limit
    unique_x = unique_x[mask]
    unique_y = unique_y[mask]

    mask_filp = abs(width-unique_x_flip)<roi_limit

    unique_x_flip = unique_x_flip[mask_filp]
    unique_y_flip = unique_y_flip[mask_filp]

    # 创建一个新的数组，大小和原图片相同，但是全为0
    new_img_array = np.full(img_array.shape,255,dtype=np.uint8)
    new_img_flip_array = np.full(img_flip_array.shape,255,dtype=np.uint8)

    # 根据坐标将新数组中对应的值设为255
    new_img_array[unique_y-1, unique_x-1] = 0
    new_img_flip_array[unique_y_flip-1, width-unique_x_flip-1] = 0
    
    if ismakescatter:

        fig=plt.scatter(unique_x-1,unique_y-1)
        plt.scatter(width-unique_x_flip-1,width-unique_y_flip-1)

        ax = fig.axes
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # 设置 x 轴向右，y 轴向下
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # 将图像保存到本地
        plt.savefig('valid_scatter.png')
        print("scatter saved in ./valid_scatter.png")
    
    return new_img_array,new_img_flip_array

if __name__ == '__main__':
    
    mask_path="/home/caojinghao/railway_object_detection/rail_mask.jpg"
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    left,right = segment_lr(img, 1400,ismakescatter=False)