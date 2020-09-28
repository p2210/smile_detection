import cv2


#face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')


#Grab Webcam Feed
webcam =  cv2.VideoCapture(0)


#Show the current frame
while True:

    #Read current frame webcam video stream
    successful_frame_read,frame = webcam.read()
    #if there's an error,abort
    if not successful_frame_read:
        break 
    #Change to grayscale
    frame_grayscale = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)   
    #Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)
    
    #Run faces detcetion within each of those faces
    for(x,y,w,h) in faces:
        #Draw a reactangle around the faces
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)
        #Get the sub face using numpy N-dimensional array slicing
        the_face = frame[y:y+h,x:x+w ]

        
        #Chnage to grayscale
        face_grayscale = cv2.cvtColor(the_face , cv2.COLOR_BGR2GRAY)
        smile = smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,minNeighbors=20)
        #Find all smiles in the faces
        #for(x_,y_,w_,h_) in smiles:
            
                    
            # draw all the rectangles around the simle
            # cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),4)
        if len(smile)>0:
            cv2.putText(frame,'smiling',(x,y+h+40),fontScale =3,fontFace = cv2.FONT_HERSHEY_PLAIN,color = (255,255,255))
    




    
      

    #show the current frame
    cv2.imshow('Why So serious?',frame)
    #display
    cv2.waitKey(1) 


#cleanup
webcam.release()
cv2.destroyAllWindows()



