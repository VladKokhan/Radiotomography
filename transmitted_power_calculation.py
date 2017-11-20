from cmath import *
import numpy as np
import matplotlib.pyplot as plt

d0 = 0.1 # minimum distance wave traveled in the media
f = 2.4e9 # frequency
omega = 2 * pi * f # circular frequency
eps1 = 8.85e-12 # air permittivity
eps2rel = 3.75 # the second media relative permittivity
eps2 = eps2rel * 8.85e-12 # the second media permittivity
sigma = 0 # dielectric material conductivity
myu = 4 * pi * 1e-7 # nonmagnetic material permeability
theta_i = 0 # incident angle
Pin = 1 # power

gamma1 = sqrt(1j * omega * myu * (sigma + 1j * omega * eps1))
gamma2 = sqrt(1j * omega * myu * (sigma + 1j * omega * eps2))
theta_t = asin(sin(theta_i) * sqrt(eps1/eps2)) # angle wave transmited in second media
etta1 = -omega * myu / (1j * gamma1) # the first media intrinsic impedance
etta2 = -omega * myu / (1j * gamma2) # the second media intrisic impedance
betta2 = omega * sqrt(myu * eps2) # the second media phase constant (wave number)

Z1 = etta1 * cos(theta_i) # the first media characteristic impedance
Z2 = etta2 * cos(theta_t) # the second media characteristic impedance
Z3 = Z1 # the third media chracteristic impedance

de = np.zeros(101, dtype=complex) # effective distance wave traveled in the media
Zin = np.zeros(101, dtype=complex) # the input impedance at the interface between the first and the second medias
PR = np.zeros(101) # reflected power
G = np.zeros(101, dtype=complex) # reflection coefficient
Pt = np.zeros(101) # transmitted power

for b in range(101): # transmitted power calculation in increments of 0.1 meter
    de[b] = d0 * b / cos(theta_t)
    Zin[b] = Z2 * (Z3 + Z2 * tanh(gamma2 * de[b])) / (Z2 + Z3 * tanh(gamma2 * de[b]))
    G[b] = (Zin[b] - Z1) / (Zin[b] + Z1) 
    PR[b] = Pin * abs(G[b])**2
    Pt[b] = Pin - PR[b]

fog = plt.figure()
plt.plot(de, Pt)
#plt.savefig('Transmitted power', fmt='png')
plt.show()
