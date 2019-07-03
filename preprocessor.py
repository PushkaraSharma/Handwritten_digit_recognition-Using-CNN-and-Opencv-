import cv2
import numpy as np


def makesquare(not_sq):
    bl=[0,0,0]
    img_dim = not_sq.shape
    h = img_dim[0]
    w = img_dim[1]
    print('h:',h,'w:',w)
    if h ==w:
        sq = not_sq
        return sq
    else:
        doubleS = cv2.resize(not_sq,(2*w,2*h),interpolation=cv2.INTER_CUBIC)
        h = 2*h
        w = 2*w
        if(h>w):
            pad = int((h-w)/2)
            doubleSS = cv2.copyMakeBorder(doubleS,0,0,pad,pad,cv2.BORDER_CONSTANT,value=bl)
        else:
            pad = int((w-h)/2)
            doubleSS = cv2.copyMakeBorder(doubleS,pad,pad,0,0,cv2.BORDER_CONSTANT,value=bl)

    doubleS_dim = doubleSS.shape
    print('height = ',doubleS_dim[0],'width = ',doubleS_dim[1])
    return (doubleSS)


def resize_pixel(dim,image):
    b_p =4
    dim = dim-b_p
    sq = np.array(image)
    r = float(dim)/sq.shape[1]
    dim = (dim,int(sq.shape[0]*r))
    resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    h_r = img_dim2[0]
    w_r = img_dim2[1]
    bl = [0,0,0]
    if(h_r>w_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=bl)
    if (h_r < w_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=bl)
    p = 2
    Resizeimg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=bl)
    img_dim = Resizeimg.shape
    h = img_dim[0]
    w = img_dim[1]
    print('h:',h,'w:',w)
    return Resizeimg
