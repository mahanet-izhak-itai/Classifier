#opens up a webcam feed so you can then test your classifer in real time
#using detectMultiScale
import numpy
import cv2

buf = []

def detect(img):
    cascade = cv2.CascadeClassifier("output/cascade.xml")
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    global buf
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        buf.append([(x1 + x2) / 2, (y1 + y2) / 2])
    if 10 < len(buf):
        buf = buf[-10:]
    for point in buf:
        cv2.circle(img,(point[0],point[1]), 5, (0,0,255), -1)


cap = cv2.VideoCapture(0)
cap.set(3,400)
cap.set(4,300)

while(True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    rects, img = detect(img)
    box(rects, img)
    cv2.imshow("frame", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
