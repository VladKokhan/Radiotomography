import numpy as np
import cv2
from cmath import *
from math import radians
import matplotlib.pyplot as plt
from minimg import MinImg, QO_SUBPIXEL
from scipy import ndimage

def dielecric_loss(P0, eps2rel, sigma2, parallel=True):
    f = 2.4e9 # frequency
    omega = 2.0 * pi * f # circular frequency
    eps1 = 8.85e-12 # the first media permittivity
    myu1 = 4.0 * pi * 1e-7 # nonmagnetic material permeability
    myu2 = 4.0 * pi * 1e-7
    sigma1 = 0.0 # the first media conductivity
    theta_i = 0.0

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
        Lm = 10.0*log10(P0)
    else:
        Lm =  10.0*log10(P0/Pt)
    return(Lm.real)

def element_loss(P0):
    l1 = dielecric_loss(P0, 2, 0.0047) #wood
    l2 = dielecric_loss(P0, 2.58, 0.0217) #chipboard
    l3 = dielecric_loss(P0, 3.75, 0.038) #brick
    l4 = dielecric_loss(P0, 5,31, 0.0326) #concrete
    l5 = dielecric_loss(P0, 1.0, 59.5e6) #copper
    return(l1, l2, l3, l4, l5)

def object_RT(P0, n):
    w = 80
    h = 80
    obj = np.zeros((h, w), np.single)

    l1 = dielecric_loss(P0, 2, 0.0047, parallel=True) #wood
    l2 = dielecric_loss(P0, 2.58, 0.0217, parallel=True) #chipboard
    l3 = dielecric_loss(P0, 3.75, 0.038, parallel=True) #brick
    l4 = dielecric_loss(P0, 5.31, 0.0326, parallel=True) #concrete
    l5 = dielecric_loss(P0, 1.0, 59.5e6, parallel=True) #copper

    if n == 1:
        cv2.rectangle(obj, (0,0), (79,79), l3, 1)
        cv2.rectangle(obj, (1,1), (78,78), l3, 1)
        cv2.rectangle(obj, (29,34), (49,44), l1, 1)
    
    if n == 2:
        cv2.circle(obj,(39,39), 38, (l3), 2)
        cv2.rectangle(obj, (21,59), (27,65), l1, -1)

    if n == 3:
        cv2.rectangle(obj, (0,0), (79,79), l3, 1)
        cv2.rectangle(obj, (1,1), (78,78), l3, 1)
        cv2.rectangle(obj, (2,2), (77,77), l3, 1)
        cv2.rectangle(obj, (3,3), (76,76), l3, 1)

        cv2.rectangle(obj, (16,15), (16,35), l5, -1)
        cv2.circle(obj,(13,65), 3, (l4), -1)
        cv2.rectangle(obj, (51,46), (65,60), l1, -1)
        cv2.rectangle(obj, (57,5), (72,25), l1, 1)
        cv2.rectangle(obj, (63,16), (71,24), l2, -1)

    if n == 4:
        cv2.rectangle(obj, (0,0), (79,79), l3, 1)
        cv2.rectangle(obj, (1,1), (78,78), l3, 1)

        cv2.rectangle(obj, (16,20), (16,40), l5, -1)
        cv2.circle(obj,(13,65), 3, (l4), -1)
        cv2.rectangle(obj, (39,2), (39,77), l3, -1)
        cv2.rectangle(obj, (57,5), (72,25), l1, 1)
        cv2.rectangle(obj, (63,16), (71,24), l2, -1)

    if n == 5:
        cv2.ellipse(obj, (39,39), (38,30), 0, 0, 360, l3, 2)
        cv2.circle(obj,(24,24), 3, (l4), -1)
        cv2.rectangle(obj, (44,19), (59,34), l1, -1)
        cv2.rectangle(obj, (44,44), (44,64), l5, -1)
    
    if n == 6:
        obj = np.zeros((15, 15), np.single)
        #cv2.rectangle(obj, (110,55), (124,69), l1, -1)
        cv2.circle(obj, (5,4), 2, (l5), -1)
        #obj[3,3] = 1

    return(obj)

def rot ():
    obj = np.zeros((180, 180), np.single )
    cv2.rectangle(obj, (40,12), (40,168), 5, -1)
    #obj[:, 0] = 5
    fig = plt.figure(1)    
    plt.imshow(obj, cmap='gray')
    plt.colorbar()
    
    img = MinImg.fromarray(obj)
    img.rotate(radians(45), out=img, quality=QO_SUBPIXEL)
    fig2 = plt.figure(2)    
    plt.imshow(obj, cmap='gray')
    plt.colorbar()
    plt.show()

def rt_beam(delta, h, w):
    beam = np.zeros((h, w), np.single)
    
    for step in range(0, w, delta):
        beam[:, step] = 1

    return(beam)


def projection(obj, delta_ang, ellips=False, delta=0):
    h, w = obj.shape[:2]
    beam = rt_beam(1, h, w)

    if ellips == False:
        projection_matrix = np.zeros((int(180/delta_ang), w), np.single)
        for n in range(int(180/delta_ang)):
            ang = n * delta_ang
            new_obj = obj.copy()
            img = MinImg.fromarray(new_obj)
            img.rotate(radians(ang), out=img, quality=QO_SUBPIXEL)
            projection_matrix[n, :] = np.sum((new_obj*beam), axis=0)

    
    elif ellips == True:
        beam_vector = np.zeros(h*3)
        projection_matrix = np.zeros((int(180/delta_ang), w), np.single)
        beam = np.zeros((h, 3))
        cv2.ellipse(beam, (1, int(h/2)), (1, int(h/2)), 0, 0, 360, 255, -1)
        sum_beam = np.sum(beam, axis=1)
        for c in range(3):
            beam[:,c] = beam[:,c] / sum_beam
        for n in range(0, 180, delta_ang):
            new_obj = obj.copy()
            img = MinImg.fromarray(new_obj)
            img.rotate(radians(n), out=img, quality=QO_SUBPIXEL)

            for step in range(0, w, delta):
                for c in range(3):
                    for r in range(h):
                        if beam[r, c] > 0:
                            if c+step < 80:
                                beam_vector[r + h * c] = new_obj[r,c + step] * beam[r,c]
                    
                projection_matrix[n, step] = np.sum(beam_vector)

    projection_matrix = np.append(projection_matrix, np.flip([projection_matrix[0]],1), axis=0)
    projection_matrix = np.delete(projection_matrix, 0, 0)            
    projection_matrix = np.flip(projection_matrix,1)
    projection_matrix = np.flip(projection_matrix,0)
    
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
    #w = sinogram.shape[1]
    #beam_length = w
    range_loss = TP_v(beam_length, P0)
    sum_sinorgam = sinogram
    
    return(sum_sinorgam.real)

def weigthing_matrix(sinogram, delta_ang, ellips=False):
    
    a, w = sinogram.shape[:2]
    #print(int((a*w * sqrt(2)).real))
    matrix_w = np.zeros((a*w , w*w))
    beam_len = int((w/sqrt(2)).real)
    beam_start = int((w - beam_len)/2)

    if ellips == False:
        for c in range(beam_len):
            angle = 0
            for ang in range(a):
                beam = np.zeros((w, w), np.single)
                
                #cv2.rectangle(beam, (beam_start+c,beam_start), (beam_start+c, beam_start+beam_len), 1, -1)
                beam[beam_start:beam_start+beam_len, beam_start+c] = 1
                img = MinImg.fromarray(beam)
                img.rotate(radians(angle*delta_ang), out=img, quality=QO_SUBPIXEL)
                matrix_w[(beam_start+c + ang*w), :] = beam.flatten()
                angle += 1

    if ellips == True:
        for c in range(beam_len):
            angle = 0
            for ang in range(a):
                beam = np.zeros((w, w), np.single)
                ell = np.zeros((beam_len,beam_len), np.single)
                cv2.ellipse(ell, (int(c), int(beam_len/2)), (1, int(beam_len/2)), 0, 0, 360, 1, -1)
                sum_beam = np.sum(ell, axis=1)
                for q in range(beam_len):
                    ell[:,q] = ell[:,q] / sum_beam
                beam[beam_start:beam_start+beam_len, beam_start:beam_start+beam_len] = ell[:,:] 
                img = MinImg.fromarray(beam)
                img.rotate(radians(angle*delta_ang), out=img, quality=QO_SUBPIXEL)
                matrix_w[(beam_start+c + ang*w), :] = beam.flatten()
                angle += 1


    return(matrix_w)

def grad_func(matrix_w, sinogram, x, lmbd=1, func='usial'):
    if func == 'usial':
        grad = 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten())
    elif func == 'l1':
        grad = 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten()) + lmbd * np.sign(x)
    elif func == 'l2':
        grad = 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten()) + lmbd * 2 * x
    elif func == 'mix':
        grad = 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten()) + lmbd * np.sign(x) + lmbd * 2 * x
    elif func == 'TV':
        grad = 2 * np.matmul(matrix_w.T, np.matmul(matrix_w, x)-sinogram.flatten()) + lmbd * np.sign(img_grad(x))
    return(grad)

def function(matrix_w, sinogram, x , lmbd=1, func='usial'):
    if func == 'usial':
        f_x = sum((np.matmul(matrix_w, x.flatten()) - sinogram.flatten())**2)
    elif func == 'l1':
        f_x = sum((np.matmul(matrix_w, x.flatten()) - sinogram.flatten())**2) + lmbd * sum((abs(x)))
    elif func == 'l2':
        f_x = sum((np.matmul(matrix_w, x.flatten()) - sinogram.flatten())**2) + lmbd * sum(((x)**2))
    elif func == 'mix':
        f_x = sum((np.matmul(matrix_w, x.flatten()) - sinogram.flatten())**2) + lmbd * sum(((x)**2)) + lmbd * sum((abs(x)))
    elif func == 'TV':
        f_x = sum((np.matmul(matrix_w, x.flatten()) - sinogram.flatten())**2) + lmbd * sum((abs(img_grad(x))))
    return(f_x)

def img_grad(x):
    hh = x.shape[0]
    h = int((sqrt(hh)).real)
    x = np.reshape(x, (h, h))

    roberts_cross_v = np.array( [[ -1, 0],[ 0, 1]] )
    roberts_cross_h = np.array( [[ 0, -1],[ 1, 0]] )

    vertical = ndimage.convolve( x, roberts_cross_v )
    horizontal = ndimage.convolve( x, roberts_cross_h )

    output = abs(horizontal) + abs(vertical)

    return(output.flatten())



def gradient_descent(matrix_w, sinogram, n):
    a, w = sinogram.shape[:2]
    h = w

    x_matrix = np.zeros((h,w))
    x = x_matrix.flatten()
    lmbd = 0.00001

    for t in range(n):
        x += - lmbd * grad_func(matrix_w, sinogram, x, 2.0, func='l1')
    return(np.reshape(x, (h,w)))

def ternary_search(matrix_w, sinogram, x, grad, left, right, eps=1e-5, f='usial'):
    while abs(right - left) > eps:
        a = (left * 2.0 + right) / 3.0
        b = (left + right * 2.0) / 3.0
        f1 = function(matrix_w, sinogram,  x - a * grad, 2.0, func=f)
        f2 = function(matrix_w, sinogram,  x - b * grad, 2.0, func=f)
        if f1 < f2:
            right = b
        else:
            left = a
    return (left + right) / 2.0

def steepest_gradient_descent(matrix_w, sinogram, n,f):
    a, w = sinogram.shape[:2]
    h = w

    x_matrix = np.zeros((h,w))
    x = x_matrix.flatten()
    lmbd =  0.0    
    #error = sqrt((np.sum(((sinogram.flatten())**2), axis=0)))/(sinogram.flatten()).shape[0]
    #error = error.real
    #eps = 1e-3
    step = 1
    descent_speed = np.zeros((n), np.single)
    alpha = np.zeros((n), np.single)
    #while error > eps:
    for t in range(n):
        alpha[step-1] = lmbd
        grad = grad_func(matrix_w, sinogram, x, 1.0, func=f)
        x += -lmbd * grad
        lmbd = ternary_search(matrix_w, sinogram, x, grad, 0.0, 10.0, f=f)
        #print(lmbd)
        error = sqrt(np.sum(((sinogram.flatten() - np.matmul(matrix_w, x.flatten()))**2), axis=0))/(sinogram.flatten()).shape[0]
        error = error.real
        print(step)
        #print(error)
        descent_speed[step-1] = error
        step += 1
    
    return(np.reshape(x, (h,w)),descent_speed, alpha)

if __name__ == "__main__":
    
    #rot()
    #exit()

    P0 = 1000
    object_number = 3

    obj = object_RT(P0, object_number)
    
    h,w= obj.shape[:2]

    
    obj_ext = np.zeros((int(w * 2), int(w * 2)), np.single)
    obj_ext[int((w/2)):w + int(w/2), int((w/2)):w + int(w/2)] = obj[:,:]

    #grad = img_grad(obj.flatten())

    
    fig1 = plt.figure(1)
    plt.imshow(obj, cmap='gray')
    plt.colorbar()

    #plt.show()
    #exit()
    
    delta_ang = 5
    '''
    beam = rt_beam(5, h, w)
    fig2 = plt.figure(2)
    plt.imshow(beam, cmap='gray')
    '''

    #beam = rt_beam(5, 151, 151)
    #cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
    #cv2.imshow("image3", beam)

    sinogram, beam = projection(obj_ext, delta_ang, ellips=False, delta=1)

    #sinogram = np.genfromtxt('sinogram.txt',dtype='float')
    
    #sum_sinogram = TP_sum(P0, sinogram, obj)
    
    #sum_sinogram = sinogram
    
    
    

    w_matrix = weigthing_matrix(sinogram, delta_ang, ellips=False)
    
    #fig4 = plt.figure(4)
    #plt.imshow(w_matrix, cmap='gray')
    '''
    img1 = gradient_descent(w_matrix,  sum_sinogram, 10)
    fig5 = plt.figure(5)    
    plt.imshow(img1, cmap='gray')
    plt.colorbar()
    print('L2_1 =', (sqrt(sum((img1.flatten() - obj.flatten())**2))).real/(obj.flatten()).shape[0])
    '''

    a,b = sinogram.shape[:2]
    
    sinogram = np.matmul(w_matrix, obj_ext.flatten())
    sinogram = np.reshape(sinogram, (a,b))
    '''
    fig2 = plt.figure(2)
    plt.imshow(sinogram, cmap='gray')
    plt.colorbar()
    plt.ylabel('Angle', fontsize=20)
    plt.xlabel('Receiver element', fontsize=20)
    '''
    it = 10

    img2, speed_1, alpha = steepest_gradient_descent(w_matrix,  sinogram, it,f='usial')
    fig6 = plt.figure(6)    
    plt.imshow(img2, cmap='gray')
    plt.colorbar()

    obj_ct = np.zeros((h,w), np.single)
    obj_ct[:,:] = obj_ext[int((w/2)):w + int(w/2), int((w/2)):w + int(w/2)]
    img_2_ct = np.zeros((h,w), np.single)
    img_2_ct[:,:] = img2[int((w/2)):w + int(w/2), int((w/2)):w + int(w/2)]
    fig7 = plt.figure(7)    
    plt.imshow(img_2_ct, cmap='gray')
    plt.colorbar()

    print('L2_1 =', (sqrt(sum((img2.flatten() - obj_ext.flatten())**2))).real/(obj_ext.flatten()).shape[0])
    print('L2_2 =', (sqrt(sum((img_2_ct.flatten() - obj_ct.flatten())**2))).real/(obj_ct.flatten()).shape[0])
    
    
    img3, speed_2, alpha = steepest_gradient_descent(w_matrix,  sinogram, it,f='TV')
    fig8 = plt.figure(8)    
    plt.imshow(img3, cmap='gray')
    plt.colorbar()
    print('L2_2 =', (sqrt(sum((img3.flatten() - obj_ext.flatten())**2))).real/(obj_ext.flatten()).shape[0])
    #plt.show()
    #exit()
    


    '''
    img4, speed_2, alpha = steepest_gradient_descent(w_matrix,  sum_sinogram, 20,f='l2')
    fig8 = plt.figure(8)    
    plt.imshow(img4, cmap='gray')
    plt.colorbar()
    print('L2_2 =', (sqrt(sum((img4.flatten() - obj.flatten())**2))).real/(obj.flatten()).shape[0])

    img5, speed_2, alpha = steepest_gradient_descent(w_matrix,  sum_sinogram, 20,f='mix')
    fig9 = plt.figure(9)    
    plt.imshow(img5, cmap='gray')
    plt.colorbar()
    print('L2_2 =', (sqrt(sum((img5.flatten() - obj.flatten())**2))).real/(obj.flatten()).shape[0])
    '''

    
    fig10 = plt.figure(10)
    speed_arg = np.arange(it)
    plt.plot(speed_arg, speed_1)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize=20)
    plt.show()
    exit()

    fig11 = plt.figure(11)
    plt.plot(speed_arg, speed_2)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize=20)
    

    '''
    fig8 = plt.figure(8)
    lmbd_arg_2 = np.arange(160)
    plt.plot(lmbd_arg_2, alpha)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Lambda', fontsize=20)
    plt.grid(True)
    plt.tick_params(labelsize=20)
    '''

    '''
    s = np.matmul(w_matrix, obj_ext.flatten())
    h,w = sinogram.shape[:2]
    fig20 = plt.figure(20)
    plt.imshow(np.reshape(s, (h,w)), cmap='gray')
    plt.colorbar()
    print(sqrt(np.sum(((sum_sinogram.flatten() - s.flatten())**2), axis=0))/(sum_sinogram.flatten()).shape[0])
    '''
    plt.show()
