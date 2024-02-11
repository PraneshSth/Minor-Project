import cv2
import numpy as np
import os 
import csv
import datetime as datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

cascadePath = cv2.data.haarcascades +"haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

# initiate id counter
id = 0

names = ['None', 'mishan','samriddha','swameep','pranesh','prabhas','nitesh','prasanna','Swopnil','peris','laxmi sir','raul'] 

# initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Open the attendance.csv file in append mode
with open('attendance.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write the header row to the CSV file
    writer.writerow(['Person', 'Timestamp', 'Status'])
    
    while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            # recognize the face using the trained model
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # map the predicted id to a name
            if (id < len(names)):
                name = names[id]
            else:
                name = "unknown"

            # display the name and confidence level on the image
            cv2.putText(
                img, 
                name, 
                (x+5,y-5), 
                font, 
                1, 
                (255,255,255), 
                2
            )

            cv2.putText(
                img, 
                "{:.2f}%".format(100 - confidence), 
                (x+5,y+h-5), 
                font, 
                1, 
                (255,255,0), 
                1
            )  

            # Write data to the attendance.csv file
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status = 'Present'
            writer.writerow([name, timestamp, status])

        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
