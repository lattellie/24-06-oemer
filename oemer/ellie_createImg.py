
import cv2
def createImg(typ:str, image, input1, input2, input3, dir = 'images/outputImg0610/',imsave=False,imshow=False):
    resize_ratio = 0.3
    def createClefSfnRestImg(image, clefs, sfns, rests, dir):
        img = image.copy()
        rectthickness = 2
        clefLabels = ['','G_clef','F_clef']
        clefColors = [(255,0,0), (66,182,245),(168,132,245)]

        sfnMap = ['','flat b','sharp #','natural']
        sfnColor = [(255,0,0),(0,0,255),(255,255,0),(250,140,200)]

        restTypeMap = ['','WH','Q','E','S','T','s','W','H']
        restLabels = ['','wholeHalf','quarter','eigth','1/16','1/32','1/64','whole','half']
        RestColors = [[204, 0, 0], [0, 204, 0], [100,255,100], [176,255,200], [204, 0, 204], [0, 204, 204], [204, 143, 0], [134, 105, 196], [85, 85, 85]]

        for i in range(len(clefs)):
            c = clefs[i]
            bbox = c.bbox
            if c._label is None:
                print(f'clef label {i} is of type None')
            else:
                lb = c._label
                img = cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]),clefColors[lb.value], rectthickness, cv2.LINE_AA)
        for j in range(1,len(clefColors)):
            img = cv2.putText(img,clefLabels[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clefColors[j], 2, cv2.LINE_AA)

        for i in range(len(sfns)):
            s = sfns[i]
            if s._label is None:
                print(f'sfn label {i} is of type None')
            else:
                bbox = s.bbox
                lb = s._label
                color = sfnColor[lb.value]
                if s.is_key:
                    print(f'label {i} is key')
                    color = (255,0,255)
                img = cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]),color, rectthickness, cv2.LINE_AA)
        for j in range(len(sfnColor)):
            img = cv2.putText(img, sfnMap[j],(30,30+(len(clefColors)+j)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sfnColor[j], 2, cv2.LINE_AA)

        for i in range(len(rests)):
            r = rests[i]
            if r._label is None:
                print(f'rest label {i} is of type None')
            else:
                bbox = r.bbox
                img = cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]), RestColors[r._label.value], rectthickness, cv2.LINE_AA)
                img = cv2.putText(img, restTypeMap[r._label.value],(bbox[2],bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RestColors[r._label.value], 2, cv2.LINE_AA)
        for j in range(len(RestColors)):
            img = cv2.putText(img, restLabels[j],(30,30+(len(clefColors)+len(sfnColor)+j)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RestColors[j], 2, cv2.LINE_AA)
        if imsave:
            print(f'saving image to {dir}clefSfnRests.jpg')
            cv2.imwrite(dir+'clefSfnRests.jpg',img)
        if imshow:
            resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
            cv2.imshow("clef sfn rests", resized_image)
            cv2.waitKey()
        return img
    
    def createStaffBarlineImg(image, staffs,barlines, dir):
        img = image.copy()
        resize_ratio = 0.3
        staffColor = [(255,0,0),(255,150,0)]
        rectthickness = 2
        barcolor = (0,255,0)
        for j in range(staffs.shape[1]):
            for i in range(staffs.shape[0]):
                s = staffs[staffs.shape[0]-1-i][j]
                img = cv2.line(img, (int(s.x_left), int(s.y_lower)), (int(s.x_right), int(s.y_lower)), staffColor[s.track], 3)
                img = cv2.line(img, (int(s.x_left), int(s.y_upper)), (int(s.x_right), int(s.y_upper)), staffColor[s.track], 3)
            disptext = 'group: '+str(s.group)
            img = cv2.putText(img, disptext , (int(s.x_left), int(s.y_upper)-40) , cv2.FONT_HERSHEY_SIMPLEX,  1, staffColor[s.track], 2, cv2.LINE_AA)
        for j in range(len(staffColor)):
                img = cv2.putText(img, 'track '+str(j), (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, staffColor[j], 2, cv2.LINE_AA)
        for i in range(len(barlines)):
                b = barlines[i]
                bbox = b.bbox
                img = cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]),barcolor, rectthickness, cv2.LINE_AA)
                img = cv2.putText(img, str(b.group), (bbox[2],bbox[3]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, barcolor, 2, cv2.LINE_AA)
        if imsave:
            print(f'saving image to {dir}staffBarline.jpg')
            cv2.imwrite(dir+'staffBarline.jpg',img)
        if imshow:
            resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
            cv2.imshow("staff barline", resized_image)
            cv2.waitKey()
        return img
    
    def createNotesImg(image, notes, dir):
        img = image.copy()
        resize_ratio = 0.3
        noteTypeMap = ['W','H','Q','E','S','T','s','t','X','x']
        labels = ['whole','half','quarter','eigth','1/16','1/32','1/64','1/3','NotDefined','NoLabel']
        sfnMap = ['','b','#','%']
        rectThickness = 2
        thickness = 1
        textScale = 0.5
        colorMap = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,179,0),(168,132,245),(107,107,107),(50,50,50)]
        for i in range(len(notes)):
            n = notes[i]
            nbox = n.bbox
            nval = 9 if n._label is None else n._label.value 
            color = colorMap[nval]
            disptext = noteTypeMap[nval]+str(n.staff_line_pos)+("." if n.has_dot else "")+sfnMap[0 if n.sfn is None else n.sfn.value]
            img = cv2.rectangle(img, (nbox[0],nbox[1]), (nbox[2],nbox[3]), color, rectThickness, cv2.LINE_AA)
            img = cv2.putText(img,disptext , (nbox[2], nbox[3]), cv2.FONT_HERSHEY_SIMPLEX,  textScale , color, thickness , cv2.LINE_AA)
        for j in range(len(labels)):
                img = cv2.putText(img, labels[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorMap[j], 2, cv2.LINE_AA)
        if imsave:
            print(f'saving image to {dir}notes.jpg')
            cv2.imwrite(dir+'notes.jpg',img)
        if imshow:
            resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
            cv2.imshow("notes", resized_image )
            cv2.waitKey()
        return img

    def createNoteSfnImg(image, notes, dir):
        imgSign = image.copy()
        resize_ratio = 0.3
        labels = ['no sign','flat(b)','sharp(#)','natural']
        sfnMap = ['','b','#','%']
        sfnColor = [(255,0,0),(0,0,255),(255,255,0),(250,140,200)]
        rectThickness = 3
        thickness = 1
        textScale = 0.5
        for i in range(len(notes)):
            n = notes[i]
            nbox = n.bbox
            if not n.sfn is None:
                    color = sfnColor[n.sfn.value]
                    imgSign = cv2.rectangle(imgSign, (nbox[0],nbox[1]), (nbox[2],nbox[3]), color, rectThickness, cv2.LINE_AA)
                    imgSign = cv2.putText(imgSign,' '+sfnMap[n.sfn.value], (nbox[2], nbox[3]), cv2.FONT_HERSHEY_SIMPLEX,  textScale , color, thickness , cv2.LINE_AA)
        for j in range(len(sfnMap)):
                imgSign = cv2.putText(imgSign , labels[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sfnColor[j], 2, cv2.LINE_AA)
        if imsave:
            print(f'saving image to {dir}notesSfn.jpg')
            cv2.imwrite(dir+'notesSfn.jpg',imgSign)
        if imshow:
            resized_image = cv2.resize(imgSign, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
            cv2.imshow("notes", resized_image )
            cv2.waitKey()
        return imgSign

    if typ.lower()=='clefsfnrest':
        img = createClefSfnRestImg(image=image, clefs=input1, sfns=input2, rests=input3, dir=dir)
        return img
    elif typ.lower() == 'staffbarline':
        img = createStaffBarlineImg(image=image, staffs=input1,barlines=input2, dir=dir)
        return img
    elif typ.lower() == 'notes':
        img = createNotesImg(image=image, notes=input1, dir=dir)
        return img
    elif typ.lower() == 'notesfn':
        img = createNoteSfnImg(image=image, notes=input1, dir=dir)
        return img
    else:
        print('type should be one of clefsfnrest/staffbarline/notes/notesfn')
        return image