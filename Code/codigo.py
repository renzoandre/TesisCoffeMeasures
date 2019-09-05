import cv2
import numpy as np
import math

# Leer imagen 
original = cv2.imread('cereza_seca.png', 1)
original = cv2.resize(original,(1536, 1024))
cv2.imshow('Original', original)

# Grises
gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
cv2.imshow('Grises', gray)

# Filtrado
#filtered = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow('Gausiana', filtered)

# Binarizacion
t, binarized = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU)
cv2.imshow('Binarizada', binarized)

# Erosion y dilatacion
mask = np.ones((7, 7), np.uint8)
imgeroded = cv2.erode(binarized, mask, iterations = 1)
imgdilated = cv2.dilate(imgeroded, mask, iterations = 1)
cv2.imshow('Apertura', imgdilated)

# Inversa
imginverted = cv2.bitwise_not(imgdilated)
cv2.imshow('Inverso', imginverted)

# Foreground
imgforeground = cv2.copyTo(original, imginverted)
cv2.imshow('Foreground', imgforeground)

# Contours
countours, hierarchy = cv2.findContours(imginverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset = (0,0))
imgcountours = cv2.drawContours(original, countours, -1, (255, 0, 0), thickness = 1)
cv2.imshow('Contours', imgcountours)

# Moments
font = cv2.FONT_HERSHEY_SIMPLEX
n = 0
centerPoints = []
for i in countours:
    moments = cv2.moments(i)
    if moments['m00'] != 0.0:
        px = int(moments['m10'] / moments['m00'])
        py = int(moments['m01'] / moments['m00'])
        cv2.circle(original, (px, py), 2, (0,255,0), -1)
        centerPoints.append([px, py])
        #if n % 2 == 0:
            #cv2.putText(original, str(n) + "-(x:" + str(px) + ",y:" + str(py) + ")", (px+5, py+15), font, 0.4, (0,0,255), 1)
            #cv2.putText(original, str(n) + "-" + str(py), (px+5, py+15), font, 0.4, (0,0,255), 1)
        #else:
            #cv2.putText(original, str(n) + "-(x:" + str(px) + ",y:" + str(py) + ")", (px+5, py-15), font, 0.4, (0,0,255), 1)
            #cv2.putText(original, str(n) + "-" + str(py), (px+5, py-15), font, 0.4, (0,0,255), 1)
        n = n + 1
cv2.imshow('Moments', original)

# Distance
pos = 0
for i in countours:
    moments = cv2.moments(i)
    if moments['m00'] != 0.0:
        shorter_dist = 10000
        longer_dist = 0
        x_short_dist = 0
        y_short_dist = 0
        x_long_dist = 0
        y_long_dist = 0
        x_center = centerPoints[pos][0]
        y_center = centerPoints[pos][1]
        for j in i:
            x_countour = j[0][0]
            y_countour = j[0][1]
            radio = math.sqrt((x_countour - x_center) ** 2 + ((y_countour - y_center) ** 2))
            if radio < shorter_dist:
                shorter_dist = radio
                x_short_dist = x_countour
                y_short_dist = y_countour
            if radio > longer_dist:
                longer_dist = radio
                x_long_dist = x_countour
                y_long_dist = y_countour
        pos = pos + 1
        cv2.line(original, (x_center, y_center), (x_short_dist, y_short_dist), (0,0,255), 1, cv2.LINE_AA)
        cv2.circle(original, (x_center, y_center), int(round(shorter_dist)) , (0,0,255), 1, cv2.LINE_AA)
        cv2.line(original, (x_center, y_center), (x_long_dist, y_long_dist), (0,0,255), 1, cv2.LINE_AA)
        cv2.circle(original, (x_center, y_center), int(round(longer_dist)) , (0,0,255), 1, cv2.LINE_AA)
        # Areas
        area_minor = math.pi * math.pow(shorter_dist, 2)
        area_major = math.pi * math.pow(longer_dist, 2)
        area_grain = cv2.contourArea(i)
        cv2.putText(original, str(pos), (x_center + 15, y_center - 15), font, 0.4, (0,0,255), 1, cv2.LINE_AA)
        print("______________________________")
        print("Grano Nro.: " + str(pos))
        print("Área menor: " + str(area_minor))
        print("Área mayor: " + str(area_major))
        print("Área grano: " + str(area_grain))
        print("% de área menor: " +  str(area_minor * 100 / area_grain))
        print("% de área mayor: " +  str(area_grain * 100 / area_major))
        # Ellipse
        if area_minor != 0.0:
            (x, y), (d_major, d_minor), angle = cv2.fitEllipse(i)
            ellipse = (x, y), (d_major, d_minor), angle
            print("Area elipse: " + str(math.pi * (d_major / 2.0) * (d_minor / 2.0)))
            cv2.ellipse(original, ellipse, (0, 255, 0), 1, cv2.LINE_AA)
cv2.imshow('Distances', original)





cv2.waitKey()
cv2.destroyAllWindows()

#cv2.imwrite('copia.png', img_grises)
        
        
