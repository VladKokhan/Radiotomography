import numpy as np
import cv2
from cmath import *
import matplotlib.pyplot as plt

def object_RT(n):
    w = h = 101
    if n == 5:
        w = 151
    obj = np.zeros((h, w), np.uint8)

    if n == 1:
        cv2.rectangle(obj, (0,0), (100,100), 150, 2)
        cv2.rectangle(obj, (40,45), (60,55), 50, 1)
    
    if n == 2:
        cv2.circle(obj,(50,50), 50, (150), 2)
        cv2.rectangle(obj, (71,29), (76,34), 50, -1)

    if n == 3:
        cv2.rectangle(obj, (0,0), (100,100), 150, 3)
        cv2.rectangle(obj, (17,16), (17,35), 250, -1)
        cv2.circle(obj,(12,88), 3, (200), -1)
        cv2.rectangle(obj, (74,69), (87,82), 50, -1)
        cv2.rectangle(obj, (80,5), (94,24), 50, 1)
        cv2.rectangle(obj, (86,16), (93,23), 100, -1)

    if n == 4:
        cv2.rectangle(obj, (0,0), (100,100), 150, 2)
        cv2.rectangle(obj, (17,21), (17,40), 250, -1)
        cv2.circle(obj,(12,88), 3, (200), -1)
        cv2.rectangle(obj, (50,2), (50,98), 150, -1)
        cv2.rectangle(obj, (80,5), (94,24), 50, 1)
        cv2.rectangle(obj, (86,16), (93,23), 100, -1)

    if n == 5:
        cv2.ellipse(obj, (75,50), (75,50), 0, 0, 360, 150, 2)
        cv2.circle(obj,(35,30), 3, (200), -1)
        cv2.rectangle(obj, (105,25), (119,39), 50, -1)
        cv2.rectangle(obj, (80,55), (80,74), 250, -1)
    
    return(obj)

def rt_beam(ang, h, w):
    x1 = 0
    y1 = 0
    x2 = w 
    y2 = h
    
    x = np.array(())
    y = np.array(())

    beam = np.zeros((h, w), np.uint8)
    new_beam = np.zeros((h, w), np.uint8)
    pts = np.array([[x1,y1],[x2,y2]], np.int32)
    cv2.polylines(beam,[pts],True,(255))
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
    beam = cv2.warpAffine(beam, rotation_matrix, (w, h))
    '''
    for c in range(h):
        for r in range(w):
            if beam[c, r] > 80:
                x = np.append(x, r)
                y = np.append(y, c)

    if ang <= 90:
        new_pts = np.array([[x1, y[0]], [x2, y[-1]]], np.int32)
    elif ang > 90 and ang < 180:
        new_pts = np.array([[x[0], y1], [x[-1], y2]], np.int32)
    cv2.polylines(new_beam,[new_pts],True,(255))
    '''
    return(beam)

def projection(obj, delta_ang, ellips=False):
    if ellips == False:
        h, w = obj.shape[:2]
        beam_vector = np.array((), np.uint8)
        range_vector = np.array(())
        x = np.array(())
        y = np.array(())

        n = range(0, 180, delta_ang)
        j = 0
        k = 1
        print(len(n))
        matrix_p = np.zeros((len(n), 210), np.uint8)        
        new_obj = obj.copy()
        for ang in n:
            beam = rt_beam(ang, h, w)
            beam_vector = np.array((), np.uint8)
            
            if ang > 45 and ang <= 225:
                beam = np.fliplr(beam)
                obj = np.fliplr(obj)
            
            for c in range(h):
                for r in range(w):
                    if beam[c, r] > 50:
                        beam_vector = np.append(beam_vector, obj[c,r])
                        x = np.append(x, r)
                        y = np.append(y, c)
            '''
            if len(beam_vector) > k:
                k = len(beam_vector)
                matrix_p.resize((len(n), k), refcheck=False)
            '''
            
            if ang > 45 and ang <= 225:
                beam_vector = beam_vector[::-1]
                beam = np.fliplr(beam)
                obj = np.fliplr(obj)
            
            for i in range(len(beam_vector)):
                matrix_p[j, i] = beam_vector[i]
            new_obj = cv2.add(new_obj, beam)
            j += 1
            range_vector = np.append(range_vector,sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2))
    return (new_obj, matrix_p, range_vector)

def TP_v(range_vector, Pin):
    vector_tp_v = np.array(())

    f = 2.4e9 # frequency
    c = 3e8 # the speed of light constant
    theta = 10
    D = 41250 / (theta**2)
    eff = 0.95 # antenna efficiency
    lmbd = c / f # mave length
    for v in range(len(range_vector)):
        Pt = ( Pin * eff**2 * D**2 * lmbd**2 / (4 * pi * range_vector[v] * 0.1)**2)
        vector_tp_v = np.append(vector_tp_v, Pt)
    return(vector_tp_v)

def TP_d(matrix_p, parallel=True):
    f = 2.4e9 # frequency
    omega = 2 * pi * f # circular frequency
    eps1 = 8.85e-12 # the first media permittivity
    myu1 = 4 * pi * 1e-7 # nonmagnetic material permeability
    myu2 = 4 * pi * 1e-7
    sigma1 = 0 # the first media conductivity
    theta_i = 0 # incident angle
    h, w = matrix_p.shape[:2]
    matrix_d = np.zeros((h, w), complex)
    for dh in range(h):      
        Pin = 1000 # power
        for dw in range(w):
            if matrix_p[dh,dw] == 50:
                eps2rel = 2
                sigma2 = 0.0047
            if matrix_p[dh,dw] == 100:
                eps2rel = 2.58
                sigma2 = 0.0217
            if matrix_p[dh,dw] == 150:
                eps2rel = 3.75
                sigma2 = 0.038
            if matrix_p[dh,dw] == 200:
                eps2rel = 5.31
                sigma2 = 0.0326
            if matrix_p[dh,dw] == 250:
                eps2rel = 1
                sigma2 = 59.5e6

            if matrix_p[dh,dw] == 0:
                Pin=Pin
                matrix_d[dh,dw] = Pin
            else:
                eps2 = eps2rel * 8.85e-12 # the second media permittivity
                gamma1 = sqrt(1j * omega * myu1 * (sigma1 + 1j * omega * eps1)) # gamma = alpha + j * betta
                gamma2 = sqrt(1j * omega * myu2 * (sigma2 + 1j * omega * eps2))
                theta_t = asin(sin(theta_i) * sqrt(eps1 * myu1 / (eps2 * myu2))) # angle wave transmited in second media
                etta1 = -omega * myu1 / (1j * gamma1) # the first media intrinsic impedance
                etta2 = -omega * myu2 / (1j * gamma2) # the second media intrisic impedance
                betta2 = omega * sqrt(myu2 * eps2 / 2 * (sqrt(1 + sigma2**2 / (eps2**2 * omega**2)) + 1)) # the second media phase constant (wave number)
                alpha2 = omega * sqrt(myu2 * eps2 / 2 * (sqrt(1 + sigma2**2 / (eps2**2 * omega**2)) - 1)) # attenuation constant

                if parallel == True:
                    Z1 = etta1 * cos(theta_i) # the first media characteristic impedance
                    Z2 = etta2 * cos(theta_t) # the second media characteristic impedance
                else:
                    Z1 = etta1 / cos(theta_i) # the first media characteristic impedance
                    Z2 = etta2 / cos(theta_t) # the second media characteristic impedance
                Z3 = Z1 # the third media chracteristic impedance
                
                d0 = 0.1
                de = d0 / cos(theta_t)
                Zin = Z2 * (Z3 +  Z2 * tanh(gamma2 * de)) / (Z2 + Z3 * tanh(gamma2 * de))
                G = (Zin - Z1) / (Zin + Z1) 
                PR = Pin * abs(G)**2
                Pt = (Pin - PR) * exp(-2 * alpha2 * de)

                Pin = Pt
                matrix_d[dh,dw] = Pin
    return(matrix_d.real)

def TP_sum(matrix_d, vector_tp_v, Pin):
    h, w = matrix_d.shape[:2]
    vector_d = np.zeros(h)
    for c in range(h):
        vector_d[c] = matrix_d[c, -1] * vector_tp_v[c] / Pin
    return(vector_d)

if __name__ == "__main__":
    Pin = 1000
    obj = object_RT(3)
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow("image1", obj)
    
    obj2, matrix_p, range_vector = projection(obj, 5)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.imshow("image2", obj2)
    cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
    cv2.imshow("image3", matrix_p)
 
    tp_v = TP_v(range_vector, Pin)
    matrix_d = TP_d(matrix_p)
    vector_TP = TP_sum(matrix_d, tp_v, Pin)
    print(vector_TP)

    fig1 = plt.figure(1)
    plt.imshow(matrix_d, vmin=0, vmax=500)
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()