import numpy as np
import cv2
from cmath import *
from math import radians
import matplotlib.pyplot as plt
from minimg import MinImg, QO_SUBPIXEL

def dielecric_loss(P0, eps2rel, sigma2, parallel=True):
    f = 2.4e9 # frequency
    omega = 2 * pi * f # circular frequency
    eps1 = 8.85e-12 # the first media permittivity
    myu1 = 4 * pi * 1e-7 # nonmagnetic material permeability
    myu2 = 4 * pi * 1e-7
    sigma1 = 0 # the first media conductivity
    theta_i = 0

    eps2 = eps2rel * 8.85e-12 # the second media permittivity
    gamma1 = sqrt(1j * omega * myu1 * (sigma1 + 1j * omega * eps1)) # gamma = alpha + j * betta
    gamma2 = sqrt(1j * omega * myu2 * (sigma2 + 1j * omega * eps2))
    if theta_i >= pi / 4 and theta_i < pi / 2:
        theta_i = pi/2 - theta_i
    if theta_i >= pi / 2 and theta_i < 3 * pi / 4:
        theta_i = theta_i - pi/2
    if theta_i >= 3 * pi / 4:
        theta_i = pi - theta_i
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
    PR = P0 * abs(G)**2
    Pt = (P0 - PR) * exp(-2 * alpha2 * de)
    if Pt ==0:
        Lm = 10*log10(P0)
    else:
        Lm =  10*log10(P0/Pt)
    return(Lm.real)

def element_loss(P0):
    l1 = dielecric_loss(P0, 2, 0.0047) #wood
    l2 = dielecric_loss(P0, 2.58, 0.0217) #chipboard
    l3 = dielecric_loss(P0, 3.75, 0.038) #brick
    l4 = dielecric_loss(P0, 5,31, 0.0326) #concrete
    l5 = dielecric_loss(P0, 1, 59.5e6) #copper
    return(l1, l2, l3, l4, l5)

def object_RT(P0, n):
    w = 161
    h = 161
    obj = np.zeros((h, w), np.uint8)

    l1 = dielecric_loss(P0, 2, 0.0047, parallel=True) #wood
    l2 = dielecric_loss(P0, 2.58, 0.0217, parallel=True) #chipboard
    l3 = dielecric_loss(P0, 3.75, 0.038, parallel=True) #brick
    l4 = dielecric_loss(P0, 5.31, 0.0326, parallel=True) #concrete
    l5 = dielecric_loss(P0, 1, 59.5e6, parallel=True) #copper

    if n == 1:
        cv2.rectangle(obj, (30,30), (130,130), l3, 1)
        cv2.rectangle(obj, (31,31), (129,129), l3, 1)
        cv2.rectangle(obj, (70,75), (90,85), l1, 1)
    
    if n == 2:
        cv2.circle(obj,(80,80), 50, (l3), 2)
        cv2.rectangle(obj, (101,59), (106,64), l1, -1)

    if n == 3:
        cv2.rectangle(obj, (30,30), (130,130), l3, 1)
        cv2.rectangle(obj, (31,31), (129,129), l3, 1)
        cv2.rectangle(obj, (32,32), (128,128), l3, 1)
        cv2.rectangle(obj, (33,33), (127,127), l3, 1)

        cv2.rectangle(obj, (47,46), (47,65), l5, -1)
        cv2.circle(obj,(44,116), 3, (l4), -1)
        cv2.rectangle(obj, (104,99), (117,112), l1, -1)
        cv2.rectangle(obj, (110,36), (124,55), l1, 1)
        cv2.rectangle(obj, (116,47), (123,54), l2, -1)

    if n == 4:
        cv2.rectangle(obj, (30,30), (130,130), l3, 1)
        cv2.rectangle(obj, (31,31), (129,129), l3, 1)
        cv2.rectangle(obj, (32,32), (128,128), l3, 1)
        cv2.rectangle(obj, (33,33), (127,127), l3, 1)

        cv2.rectangle(obj, (47,61), (47,70), l5, -1)
        cv2.circle(obj,(44,116), 3, (l4), -1)
        cv2.rectangle(obj, (80,32), (80,128), l3, -1)
        cv2.rectangle(obj, (110,35), (124,54), l1, 1)
        cv2.rectangle(obj, (116,47), (123,54), l2, -1)

    if n == 5:
        cv2.ellipse(obj, (80,80), (75,50), 0, 0, 360, l3, 2)
        cv2.circle(obj,(40,60), 3, (l4), -1)
        cv2.rectangle(obj, (110,55), (124,69), l1, -1)
        cv2.rectangle(obj, (85,85), (85,104), l5, -1)
    
    if n ==6:
        obj = np.zeros((15, 15), np.uint8)
        cv2.circle(obj, (6,7), 2, (l5), -1)

    return(obj)

def rt_beam(delta, h, w):
    x1 = 0
    y1 = 0
    x2 = 0 
    y2 = h

    beam = np.zeros((h, w), np.uint8)
    
    for step in range(0, w, delta):
        pts = np.array([[x1 + step, y1],[x2 + step, y2]], np.int32)
        cv2.polylines(beam,[pts],True,(255))

    '''
    beam = np.zeros((h, 3), np.uint8)
    for step in range(0, w, delta):
        cv2.ellipse(beam, (1, int(h/2)), (1, int(h/2)), 0, 0, 360, 255, -1)
    '''
    # rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
    # beam = cv2.warpAffine(beam, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    #from minimg import MinImg, QO_SUBPIXEL
    #img = MinImg.fromarray(beam)
    #img.rotate(radians(ang), out=img, quality=QO_SUBPIXEL)
    #print(img)
    return(beam)


def projection(obj, delta_ang, object_number, ellips=False, delta=0):
    h, w = obj.shape[:2]
    if ellips == False:
        beam = rt_beam(1, h, w)
        projection_matrix = np.zeros((int(180/delta_ang), w))
        beam_vector = np.zeros(h)
        ang = 0
        for n in range(0, 180, delta_ang):
            new_obj = object_RT(P0, object_number)
            img = MinImg.fromarray(new_obj)
            img.rotate(radians(n), out=img, quality=QO_SUBPIXEL)

            for c in range(w):
                for r in range(h):
                    if beam[r, c] > 0:
                        beam_vector[r] = new_obj[r,c]
                projection_matrix[n, c] = np.sum(beam_vector)

    
    elif ellips == True:
        beam_vector = np.zeros(h*3)
        projection_matrix = np.zeros((int(180/delta_ang), w))
        beam = np.zeros((h, 3))
        cv2.ellipse(beam, (1, int(h/2)), (1, int(h/2)), 0, 0, 360, 255, -1)
        sum_beam = np.sum(beam, axis=1)
        for c in range(3):
            beam[:,c] = beam[:,c] / sum_beam
        for n in range(0, 180, delta_ang):
            new_obj = object_RT(P0, object_number)
            img = MinImg.fromarray(new_obj)
            img.rotate(radians(n), out=img, quality=QO_SUBPIXEL)

            for step in range(0, w, delta):
                for c in range(3):
                    for r in range(h):
                        if beam[r, c] > 0:
                            if c+step < 161:
                                beam_vector[r + h * c] = new_obj[r,c + step] * beam[r,c]
                    
                projection_matrix[n, step] = np.sum(beam_vector)

    return (projection_matrix, beam)

def TP_v(beam_length, P0):
    vector_tp_v = np.array(())

    f = 2.4e9 # frequency
    c = 3e8 # the speed of light constant
    theta = 10
    D = 41250 / (theta**2)
    eff = 0.95 # antenna efficiency
    lmbd = c / f # mave length
    Pt = ( P0 * eff**2 * D**2 * lmbd**2 / (4 * pi * beam_length * 0.1)**2)
    Lv =  10*log10(P0/Pt)
    return(Lv.real)


def TP_sum(P0, sinogram, obj):
    beam_length, w = obj.shape[:2]
    range_loss = TP_v(beam_length, P0)
    sum_sinorgam = sinogram + range_loss
    
    return(sum_sinorgam)

def weigthing_matrix(sinogram, obj, ellips=False):
    a, w = sinogram.shape[:2]
    h, w = obj.shape[:2]
    matrix_w = np.zeros((a*w , h*w))
    if ellips == False:
        for c in range(w):
            for angle in range(a):
                
                beam = np.zeros((h, w), np.single)
                beam[:,0 + c] = 1
                img = MinImg.fromarray(beam)
                img.rotate(radians(angle), out=img, quality=QO_SUBPIXEL)
                matrix_w[(c+angle*w), :] = beam.flatten()
        return(matrix_w)

def gradient_descent(matrix_w, sinogram, n):
    a, w = sinogram.shape[:2]
    h = w
    matrix_w = np.flip(matrix_w,0)
    x_matrix = np.zeros((h,w))
    x = x_matrix.flatten()
    
    grad = lambda x: 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten())

    for t in range(n):
        x = x - 1*grad(x)
    
    return(np.reshape(x, (h,w)))

def steepest_gradient_descent(matrix_w, sinogram, n):
    a, w = sinogram.shape[:2]
    h = w
    matrix_w = np.flip(matrix_w,0)
    x_matrix = np.zeros((h,w))
    x = x_matrix.flatten()
    eps = 0.0001
    grad = lambda x: 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten())
    #previous_step_size = np.sum(x, axis=0)
    lmbd = 0.1
    #while previous_step_size > eps:
    for t in range(n):
        prev_x = x
        x = prev_x - lmbd*grad(prev_x)
        #previous_step_size = np.sum(abs(x - prev_x), axis=0)
        #lmbd = np.argmin(x - lmbd * grad)
    
    return(np.reshape(x, (h,w)))

if __name__ == "__main__":
    
    P0 = 1000
    object_number = 6
    obj = object_RT(P0, object_number)
    #fig1 = plt.figure(1)
    plt.imshow(obj, vmax=5, cmap='gray')
    
    #beam = rt_beam(5, 151, 151)
    #cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
    #cv2.imshow("image3", beam)

    sinogram, beam = projection(obj, 1, object_number, ellips=False, delta=1)
    
    sum_sinogram = TP_sum(P0, sinogram, obj,)
    '''
    fig2 = plt.figure(2)
    plt.imshow(sum_sinogram, cmap='gray')
    plt.colorbar()
    plt.ylabel('Angle', fontsize=20)
    plt.xlabel('Receiver element', fontsize=20)
    '''

    w_matrix = weigthing_matrix(sinogram, obj, ellips=False)
    '''
    fig4 = plt.figure(4)
    plt.imshow(w_matrix, cmap='gray')
    '''
    
    img1 = gradient_descent(w_matrix,  sum_sinogram, 1)
    fig5 = plt.figure(5)    
    plt.imshow(img1, cmap='gray')

    img2 = gradient_descent(w_matrix,  sum_sinogram, 2)
    fig6 = plt.figure(6)    
    plt.imshow(img2, cmap='gray')

    img3 = gradient_descent(w_matrix,  sum_sinogram, 3)
    fig7 = plt.figure(7)    
    plt.imshow(img3, cmap='gray')

    img4 = gradient_descent(w_matrix,  sum_sinogram, 4)
    fig8 = plt.figure(8)    
    plt.imshow(img4, cmap='gray')

    img5 = gradient_descent(w_matrix,  sum_sinogram, 5)
    fig9 = plt.figure(9)    
    plt.imshow(img5, cmap='gray')
    
    
    plt.show()
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
