import cv2

def showRectOfType(image, rectList, name="file.jpg" ,resize_ratio = 0.35, outline_color = (0,255,0), thickness = 2):
    resized_image = image.copy()
    for objType in rectList:
        bboxee = objType.bbox
        cv2.rectangle(resized_image, (bboxee[0], bboxee[1]), (bboxee[2], bboxee[3]), outline_color, thickness)
    cv2.imwrite(name, resized_image)
    resized_image = cv2.resize(resized_image, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
    cv2.imshow("rected", resized_image)
    cv2.waitKey()
    return resized_image