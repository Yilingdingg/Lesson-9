import cv2
import os
import numpy

#https://pyimagesearch.com/2021/05/03/face-recognition-with-local-binary-patterns-lbps-and-opencv/

face_xml="haarcascade_frontalface_default.xml"
datasets="datasets"

#recognition starting
print("The face recongnition will start soon, please be in sufficient light.")
# Create a list of images and a list of corresponding names
(images, labels, names, id)=([],[],{},0)
print(os.walk(datasets))

for subdir, dirs, files in os.walk(datasets):
    for sub in dirs:
        #print(sub)
        names[id]=sub
        subpath=os.path.join(datasets, sub)
        for file in os.listdir(subpath):
            path=subpath+'/'+file
            label=id
            images.append(cv2.imread(path,0))
            labels.append(label)
        id+=1
print(images)
print(labels)
(images,labels)=[numpy.array(lis) for lis in [images,labels]]
#training the model
recogniser=cv2.face.LBPHFaceRecognizer_create()
recogniser.train(images, labels)

#detect face
face_detection=cv2.CascadeClassifier(face_xml)
#open camera
webcam=cv2.VideoCapture(0)

while True:
    ret, img=webcam.read()
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_detection.detectMultiScale(gray_img, 1.3, 4)
    print(face)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y),(x+w,y+h),(156,205,255),3)
        person=gray_img[y:y+h,x:x+h]
        person_resize=cv2.resize(person,(130,100))
        #recognise face
        prediction=recogniser.predict(person_resize)
        print(prediction)
        if prediction[1]>60:
            cv2.putText(img,f'{names[prediction[0]]} - {prediction[1]}', (x+10,y-20), cv2.FONT_ITALIC, 1, (156,205,255))
        else:
            cv2.putText(img, "Face not recognised...",(x+10,y-20), cv2.FONT_ITALIC, 1, (156,205,255))

    cv2.imshow("Webcam", img)
    key=cv2.waitKey(10)
    if key==27:
        break
