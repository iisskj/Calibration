import cv2
import numpy as np
import os
from PIL import Image

img_path = [
    'neuvsnap_0315_195015.jpg', 
    'neuvsnap_0315_195040.jpg', 
    'neuvsnap_0315_195758.jpg', 
    'neuvsnap_0315_195823.jpg', 
    'neuvsnap_0315_200118.jpg'

]
pcd_path = [
    'neuvsnap_0315_195015.png', 
    'neuvsnap_0315_195040.png', 
    'neuvsnap_0315_195758.png', 
    'neuvsnap_0315_195823.png', 
    'neuvsnap_0315_200118.png'

]

def show_img():
    pcd_root = './data/png' 
    img_root = './data/images'

    pcd_path_list = [os.path.join(pcd_root, pcd) for pcd in pcd_path]
    img_path_list = [os.path.join(img_root, img) for img in img_path]

    pcd_list = [np.array(Image.open(f)) for f in pcd_path_list]
    img_list = [np.array(Image.open(f)) for f in img_path_list]
    imgs = np.hstack(img_list)
    pcds = np.hstack(pcd_list)
    print(imgs.shape,'---',pcds.shape)
    # cv2.namedWindow('calibration', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('imgs', imgs)
    # cv2.waitKey(0)
    # cv2.imshow('pcds', pcds)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    Image.fromarray(pcds).save('pcds_image.png')
    Image.fromarray(imgs).save('imgs_image.png')

    # pcd = Image.open('pcds_image.png')
    # img = Image.open('imgs_image.png')
    # image = np.vstack([np.array(img), cv2.resize(np.array(pcd), dsize=(1080, 9600, 3), interpolation=cv2.INt)])
    # Image.fromarray(image).save('image.png')
if __name__ == '__main__':
    show_img()