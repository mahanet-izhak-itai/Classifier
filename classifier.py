#opens up a webcam feed so you can then test your classifer in real time
#using detectMultiScale
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, filters

# buf = np.array([])
buf = []
THRESHOLD = 100

def smooth_line(line):
    arr = np.array(line)

    x, y = arr.T
    t = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, 100)

    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    sigma = 10
    x3 = gaussian_filter1d(x2, sigma)
    y3 = gaussian_filter1d(y2, sigma)

    x4 = np.interp(t, t2, x3)
    y4 = np.interp(t, t2, y3)

    x4 = x4.astype(int)
    y4 = y4.astype(int)

    return np.array([x4, y4]).T

def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def list_distance(v1, v2):
    return distance(np.array(v1), np.array(v2))

def smooth_movement(path):
    new_path = filters.median_filter(path, 2)
    return list(smooth_line(np.array(path)))

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
        # cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        buf.append(center)

    if 20 < len(buf):
        buf = smooth_movement(buf)

    if 20 < len(buf):
        buf = buf[-20:]

    new_img = np.zeros((400, 300, 3), np.uint8)

    for point in buf:
        cv2.circle(new_img,(point[0],point[1]), 8, (255, 255 ,255), -1)

    return new_img

cap = cv2.VideoCapture(0)
cap.set(3,400)
cap.set(4,300)

while(True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    rects, img = detect(img)
    img = box(rects, img)
    cv2.imshow("frame", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
