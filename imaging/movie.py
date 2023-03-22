import os
import cv2
import re

'''
compress a seris of images in certain folder into a video
author Jinghao Cao 
'''

def func(name):

    number = re.findall(r'\d+', name)
    return int(number[0])

def makemovie(image_folder,fps,output):
# 视频帧率

    # 获取图片文件名列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=func)
    # 按文件名排序

    # 获取第一张图片的尺寸
    img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, channels = img.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = output
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 逐一读入图片并写入视频文件
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        out.write(img)

    # 释放资源
    out.release()

if __name__ == '__main__':
    # 要合成视频的图片所在的文件夹
    image_folder = '/home/caojinghao/rail_tracking_and_obj_detection/short_focal_results'
    output ='short_output.mp4'
    fps =10
    makemovie(image_folder,fps,output)
