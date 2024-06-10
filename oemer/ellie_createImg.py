
import cv2
def createImg(inputValue, typ: str, image, resize_ratio = 0.3,dir = "images/outputImg0610/"):
    def saveStaffImg(inputValue, image, resize_ratio,dir):
        staffs = inputValue
        img = image.copy()
        for j in range(staffs.shape[1]):
            color = (j*25,255-j*25,0)
            for i in range(staffs.shape[0]):
                s = staffs[i][j]
                img = cv2.line(img, (s.x_left, s.y_lower), (s.x_right, s.y_lower), color, 3)
                disptext = 't:'+str(s.track)+' g:'+str(s.group)
                img = cv2.putText(img, disptext , (s.x_left, int(s.y_center)) , cv2.FONT_HERSHEY_SIMPLEX,  1, color, 4, cv2.LINE_AA)
                resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
        cv2.imshow("line", resized_image)
        cv2.waitKey()
        cv2.imwrite(dir+'staff.jpg',img)
        return img
    
    if typ == 'staff':
        print('dir: ', dir)
        saveStaffImg(inputValue, image, resize_ratio,dir)
    else:
        print('invalid type, type should be one of: \n staff, ')