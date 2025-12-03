import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load known faces
path = 'faces'
images = []
names = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Attendance logging
def markAttendance(name):
    df = pd.read_csv('database/attendance.csv')
    if name not in df['Name'].values:
        time_now = datetime.now().strftime('%H:%M:%S')
        df = pd.concat([df, pd.DataFrame([[name, time_now]], columns=['Name','Time'])], ignore_index=True)
        df.to_csv('database/attendance.csv', index=False)

# Video Capture
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = names[matchIndex].upper()
                    markAttendance(name)

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize attendance CSV if not exists
    if not os.path.exists('database/attendance.csv'):
        pd.DataFrame(columns=['Name','Time']).to_csv('database/attendance.csv', index=False)
    app.run(debug=True)
