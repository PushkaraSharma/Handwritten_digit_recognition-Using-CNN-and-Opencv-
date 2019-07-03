import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from preprocessor import makesquare,resize_pixel

(x_train,y_train),(x_test,y_test) = mnist.load_data()

classifier = load_model('digitsCNN.h5')

def draw(pred,real_img):
    black = [0,0,0]
    ex_img = cv2.copyMakeBorder(real_img,0,0,0,imgL.shape[0],cv2.BORDER_CONSTANT,value=black)
    ex_img = cv2.cvtColor(ex_img,cv2.COLOR_GRAY2BGR)
    cv2.putText(ex_img,str(pred),(150,72),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),2)
    cv2.imshow('predictions',ex_img)


#for i in range(20):
#    r_n = np.random.randint(0, len(x_test))
#    img = x_test[r_n]
#    imgL = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

#    img = np.reshape(img, (1, 28, 28, 1))
#    pred = classifier.predict_classes(img)[0]
#    draw(pred, imgL)
#    cv2.waitKey(0)

#cv2.destroyAllWindows()

#on real image
image = cv2.imread('numbers3.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',gray)
cv2.waitKey(0)
blur = cv2.GaussianBlur(gray,(5,5),150)
cv2.imshow('blurred',blur)
cv2.waitKey(0)

ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('thres',thresh)

kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)
bg = cv2.dilate(closing,kernel,iterations=1)
cv2.imshow('d',bg)

canny = cv2.Canny(bg,20,150)
cv2.imshow('canny',canny)
cv2.waitKey(0)


contours, hierarchy=cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("contours number:"+str(len(contours)))


def get_contour_order(contour,cols):
    tol_fac = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1]//tol_fac)*tol_fac)*cols+origin[0]

def sor(cnts):
    i=0
    reverse = False
    bb = [cv2.boundingRect(c) for c in cnts]
    (cnts,bb) = zip(*sorted(zip(cnts,bb),key=lambda b:b[1][i],reverse=reverse))
    return (cnts,bb)

(contours,bb) = sor(contours)

#contours.sort(key=lambda x:get_contour_order(x,image.shape[1]))
#contours = sorted(contours,key=lambda c:cv2.boundingRect(c)[0]+cv2.boundingRect(c)[1]*image.shape[1])

full_n = []


for c in contours:

    cv2.drawContours(image,contours,-1,(255,0,0),3)
    cv2.imshow('con',image)
    cv2.waitKey(0)

    (x,y,w,h) = cv2.boundingRect(c)
    if w>=8 and h >= 30:
        roi = blur[y:y+h,x:x+w]
        ret, roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
        print(roi)
        roi = makesquare(roi)
        print(roi.shape)
        roi = resize_pixel(28,roi)
        cv2.imshow('oo',roi)
        cv2.waitKey(0)
        roi = roi/255
        roi = roi.reshape(1,28,28,1)

        res = str(classifier.predict_classes(roi)[0])
        full_n.append(res)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(image,res,(x,y+50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        #cv2.imshow('final',image)
        #cv2.waitKey(0)

cv2.destroyAllWindows()
print('The number is:'+''.join(full_n))