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
imgcountours = cv2.drawContours(original, countours, -1, (255, 0, 0), thickness = 1)#red
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
        cv2.circle(original, (px, py), 2, (0,255,0), -1)#green
        centerPoints.append([px, py])
        #if n % 2 == 0:
            #cv2.putText(original, str(n) + "-(x:" + str(px) + ",y:" + str(py) + ")", (px+5, py+15), font, 0.4, (0,0,255), 1)
            #cv2.putText(original, str(n) + "-" + str(py), (px+5, py+15), font, 0.4, (0,0,255), 1)
        #else:
            #cv2.putText(original, str(n) + "-(x:" + str(px) + ",y:" + str(py) + ")", (px+5, py-15), font, 0.4, (0,0,255), 1)
            #cv2.putText(original, str(n) + "-" + str(py), (px+5, py-15), font, 0.4, (0,0,255), 1)
        n = n + 1
cv2.imshow('Moments', original)

# Distancias
original_circle_minor = np.copy(original)
original_circle_major = np.copy(original)
original_ellipse = np.copy(original)
pos = 0
for c in countours:
    moments = cv2.moments(c)
    if moments['m00'] != 0.0:
        # Mayor y menor distancia
        shorter_dist = 10000
        longer_dist = 0
        x_short_dist = 0
        y_short_dist = 0
        x_long_dist = 0
        y_long_dist = 0
        x_center = centerPoints[pos][0]
        y_center = centerPoints[pos][1]
        for j in c:
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
        
        # Dibujar contornos
        cv2.line(original, (x_center, y_center), (x_short_dist, y_short_dist), (0,255,0), 1, cv2.LINE_AA)#red
        cv2.circle(original, (x_center, y_center), int(round(shorter_dist)) , (0,0,255), 1, cv2.LINE_AA)#red
        cv2.line(original, (x_center, y_center), (x_long_dist, y_long_dist), (0,255,0), 1, cv2.LINE_AA)#red
        cv2.circle(original, (x_center, y_center), int(round(longer_dist)) , (0,0,255), 1, cv2.LINE_AA)#red
        cv2.putText(original, str(pos), (x_center + 15, y_center - 15), font, 0.4, (0,255,0), 1, cv2.LINE_AA)#red
        
        # Segmentos de imagen
        if shorter_dist > 0.0:
            # Segmento grano
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 1)#blue
            img_cut_grain = imgdilated[y: y + h, x: x + w]
            path_image = "segmentos/%d_img_grain.png" %pos
            #cv2.imwrite(path_image, img_cut_grain)
            
            # Segmento círculo pequeño
            cv2.circle(original_circle_minor, (x_center, y_center), int(round(shorter_dist)) , (0,0,0), -1, cv2.LINE_AA)#red
            gray_c_mi = cv2.cvtColor(original_circle_minor, cv2.COLOR_RGB2GRAY)
            value = 254
            gray_new = np.where((255 - gray_c_mi) < value, 255, gray_c_mi + value)
            tt, binarized_cut = cv2.threshold(gray_new, 200, 255, cv2.THRESH_OTSU)
            img_cut_circle_minor = binarized_cut[y: y + h, x: x + w]
            path_image = "segmentos/%d_img_circle_minor.png" %pos
            #cv2.imwrite(path_image, img_cut_circle_minor)
            
            # Segmento círculo grande
            cv2.circle(original_circle_major, (x_center, y_center), int(round(longer_dist)) , (0,0,0), -1, cv2.LINE_AA)#red
            gray_c_ma = cv2.cvtColor(original_circle_major, cv2.COLOR_RGB2GRAY)
            tt, binarized_cut = cv2.threshold(gray_c_ma, 200, 255, cv2.THRESH_OTSU)
            img_cut_circle_major = binarized_cut[y: y + h, x: x + w]
            path_image = "segmentos/%d_img_circle_major.png" %pos
            #cv2.imwrite(path_image, img_cut_circle_major)
            
            # Ellipse
            (x_e, y_e), (d_major, d_minor), angle = cv2.fitEllipse(c)
            ellipse = (x_e, y_e), (d_major, d_minor), angle
            cv2.ellipse(original, ellipse, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.ellipse(original_ellipse, ellipse, (0, 0, 0), -1, cv2.LINE_AA)
            gray_e = cv2.cvtColor(original_ellipse, cv2.COLOR_RGB2GRAY)
            tt, binarized_cut = cv2.threshold(gray_e, 200, 255, cv2.THRESH_OTSU)
            img_cut_ellipse = binarized_cut[y: y + h, x: x + w]
            path_image = "segmentos/%d_img_ellipse.png" %pos
            #cv2.imwrite(path_image, img_cut_ellipse)
            
            # Comparacion de puntos de coincidencia
            rows, cols = img_cut_grain.shape
            total = rows * cols
            num_circle_minor = 0
            num_circle_major = 0
            num_ellipse = 0
            for i in range(rows):
                for j in range(cols):
                    if img_cut_grain[i, j] == img_cut_circle_minor[i, j]:
                        num_circle_minor += 1
                    if img_cut_grain[i, j] == img_cut_circle_major[i, j]:
                        num_circle_major += 1
                    if img_cut_grain[i, j] == img_cut_ellipse[i, j]:
                        num_ellipse += 1
            perc_circle_minor = 100 * num_circle_minor / total
            perc_circle_major = 100 * num_circle_major / total
            perc_ellipse = 100 * num_ellipse / total
            
            # Areas
            area_minor = math.pi * math.pow(shorter_dist, 2)
            area_major = math.pi * math.pow(longer_dist, 2)
            area_ellipse = math.pi * (d_major / 2.0) * (d_minor / 2.0)
            area_grain = cv2.contourArea(c)
            print("______________________________")
            print("Grano Nro.: " + str(pos))
            print("Área menor: " + str(area_minor))
            print("Área mayor: " + str(area_major))
            print("Area elipse: " + str(area_ellipse))
            print("Área grano: " + str(area_grain))
            print("% de área menor: " +  str(area_minor * 100 / area_grain))
            print("% puntos circulo menor: "  + str(perc_circle_minor))
            print("% de área mayor: " +  str(area_grain * 100 / area_major))
            print("% puntos circulo mayor: "  + str(perc_circle_major))
            print("% de área elipse: " +  str(area_grain * 100 / area_ellipse))
            print("% puntos elipse: "  + str(perc_ellipse))

cv2.imshow('Distances', original)





cv2.waitKey()
cv2.destroyAllWindows()

#cv2.imwrite('copia.png', img_grises)
        
        
