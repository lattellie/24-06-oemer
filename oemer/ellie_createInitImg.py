import cv2
import numpy as np

def createInitImg(staff, symbols, stems_rests, notehead, clefs_keys, image, imshow=False, dir='images/testing/mergeImg/'):
    img=image.copy()
    resize_ratio = 0.3
    img = img//10+230
    color1 = (255,0,255) #margenta
    color2 = (255,0,0) #blue
    color3 = (255,255,0) #cyan
    colors = [color1, color2, color1, color2, color1]

    maplabel = ('staff','symbols','stem_rests','notehead','clefs_keys')
    a = (staff, symbols, stems_rests, notehead, clefs_keys)
    img0 = img.copy()
    for i in range(len(a)):
            idx = np.where(a[i]==1)
            img1 = img.copy()		
            img1 = cv2.putText(img1, maplabel[i], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,color3, 4, cv2.LINE_AA)
            img1[idx[0],idx[1]] = colors[i]
            img0 = np.hstack((img0,img1))
    if imshow:
        resized_image = cv2.resize(img0, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
        cv2.imshow("rected", resized_image)
        cv2.waitKey()
    cv2.imwrite(dir+'_initImg.jpg',img0)
    print(f'img saved at {dir}_initImg.jpg')
    return img0