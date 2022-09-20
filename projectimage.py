import cv2
import numpy as np
import face_recognition as fc
i=1
fd=cv2.CascadeClassifier("C:/Users/kiit/OneDrive - Brandscapes Consultancy Pvt. Ltd/Desktop/L&B image processing/haarcascade_frontalface_default.xml")
v=cv2.VideoCapture(0)
while True:
    _,image=v.read()
    image=np.flip(image,1)
    image=cv2.resize(image,(600,600))
    D_face_edges=fd.detectMultiScale(image,1.3,7)
    #print(len(D_face_edges))
    if len(D_face_edges)>0:
        pic=image
        for (x,y,w,h) in D_face_edges:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        f=fc.load_image_file(r"C:/Users/kiit/OneDrive - Brandscapes Consultancy Pvt. Ltd/Desktop/Video resume and resumes MBA files/2824.jpg")
        my_face_f=fc.face_encodings(f)[0]
        test_image_f=fc.face_encodings(image)[0]
        result=fc.compare_faces([test_image_f],my_face_f)
        print(result)
        if (int(result[0])==0):
            cv2.imwrite("person53{}.jpg".format(i),image)
            i=i+2
        
        
            
    #image=np.flip(image,1)
    key=cv2.waitKey(1)
    if (key==ord('q')):
        cv2.destroyAllWindows()
        v.release()
        break
    cv2.imshow("Detected face",image)
    
    
    
