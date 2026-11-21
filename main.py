from PIL import Image

import numpy as np
import math
import os



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg

im_path = r"D:/luowei_temp/SRCNN_try1/Super-Resolution_CNN-master/dataset1/pngdata/LR_data/"
out_path = r'D:/luowei_temp/SRCNN_try1/Super-Resolution_CNN-master/dataset1/pngdata/LR_biclPNG/'  #事先建立好文件夹
files = os.listdir(im_path)
for spl_file in files:
    file_name = spl_file.strip('D:/luowei_temp/SRCNN_try1/Super-Resolution_CNN-master/dataset1/pngdata/LR_data,.')
    print('--print-bicl-:',file_name)

    image = np.array((Image.open(im_path + spl_file)))
    image3 = BiCubic_interpolation(image, image.shape[0] * 4, image.shape[1] * 4)    # 4 为放大倍数  可改
    # 因为图片在imread过程中，cv2读取的结果图片形式为BRG 需要转化RGB
    image3 = Image.fromarray(image3.astype('uint8')).convert('RGB')
    image3.save(out_path + file_name+'.png')
