import cv2
import numpy as np

def yuvar(image):
    gri = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gri, (5, 5), 0)

    yuvar = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    if yuvar is not None:
        yuvar = np.uint16(np.around(yuvar))
        for i in yuvar[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)

    return image

kam = cv2.VideoCapture(0)

while True:
    ret, frame = kam.read()
    recap = cv2.flip(frame ,1)
    
    belirli = yuvar(recap)

    cv2.imshow('Circle Detection', belirli)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kam.release()
cv2.destroyAllWindows()
