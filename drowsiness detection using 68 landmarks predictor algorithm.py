import imutils
import dlib
from scipy.spatial import distance as dist
import winsound
from imutils import face_utils
import cv2

frequency=2500
duration=2000

def eyeaspectratio(eyecoordinate):
    a_coordinate=dist.euclidean(eyecoordinate[1],eyecoordinate[5])
    b_coordinate=dist.euclidean(eyecoordinate[2],eyecoordinate[4])
    c_coordinate=dist.euclidean(eyecoordinate[0],eyecoordinate[3])
    eyeaspect=(a_coordinate+b_coordinate)/(2.0*c_coordinate)
    return eyeaspect

countframe=0
eyeaspectthreshold=0.3
eyeapectframes=48
landmarkshapepredictor="shape_predictor_68_face_landmarks.dat"

camera=cv2.VideoCapture(0)
facedetector=dlib.get_frontal_face_detector()
shapepredictor=dlib.shape_predictor(landmarkshapepredictor)

(lefteyestart,lefteyeend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(righteyestart,righteyeend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    a,frame=camera.read()
    frame=imutils.resize(frame,width=450)
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rectangle=facedetector(grayscale)
    for rects in rectangle:
        shape=shapepredictor(grayscale,rects)
        shape=face_utils.shape_to_np(shape)
        lefteye=shape[lefteyestart:lefteyeend]
        righteye=shape[righteyestart:righteyeend]
        lefteyevalue=eyeaspectratio(lefteye)
        righteyevalue=eyeaspectratio(righteye)
        eyeratio=(lefteyevalue+righteyevalue)/2.0

        lefteyehull=cv2.convexHull(lefteye)
        righteyehull=cv2.convexHull(righteye)

        cv2.drawContours(frame,[lefteyehull],-1,(0,1,255),1)
        cv2.drawContours(frame,[righteyehull],-1,(0,1,255),1)

        if eyeratio<eyeaspectthreshold:
            countframe+=1
            if countframe>=eyeapectframes:
                cv2.putText(frame,"DROWSINESS DETECTED",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                winsound.Beep(frequency,duration)
        else:
            countframe=0
    cv2.imshow("Live Stream Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()




        
