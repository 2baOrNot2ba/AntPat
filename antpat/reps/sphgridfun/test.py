#!/usr/bin/python
import sys
import math
import numpy as np
import pntsonsphere
import tvecfun


def genDummyField():
    thetamsh, phimsh = pntsonsphere.sphericalGrid()
    unv = np.ones(thetamsh.shape)
    discntF = tvecfun.TVecFields(thetamsh, phimsh, unv, unv) #This field is discontinous at poles!
    return discntF

def testrotToFrame():
    rotang = math.pi/4
    rotMat = np.array([[ np.cos(rotang), -np.sin(rotang), 0.],
                     [ np.sin(rotang),  np.cos(rotang), 0.],
                     [             0.,              0., 1.]])
    theta_from = np.array([1.0, 2.0])
    phi_from = np.array([0.0, 1.0])
    theta, phi = pntsonsphere.rotToFrame(rotMat, theta_from, phi_from)
    print(theta, phi)

def testgetSphBasis():
    rvmheader =     ["x",          "y",           "z",         "+xy"]
    rvm = np.array([[1., 1e-9, 0.],[1e-9, 1., 0.],[1e-9,0.,1.],[1.,1.,0.]])
    #print rvm.shape
    #s2c = getSph2CartTransfMat(rvm)
    s2c = tvecfun.getSph2CartTransfMatT(rvm.T)
    #print s2c.shape
    for i in range(s2c.shape[0]):
        print(rvmheader[i])
        print(s2c[i])
        print("")

def printsphmsh():
    thetamsh, phimsh = pntsonsphere.sphericalGrid(2,4)
    print("Thetas:")
    print(thetamsh)
    print("Phis:")
    print(phimsh)

def testrot3Dmat():
    rotmat = pntsonsphere.rot3Dmat(0., -3.1415/2, 3.1415/2)
    print(rotmat)

def testplotFEKO(filename, request=None):
    tvf = tvecfun.TVecFields()
    tvf.load_ffe(filename, request)
    freqs = tvf.getRs()
    freq = freqs[0]
    (THETA, PHI, E_th, E_ph) = (tvf.getthetas(), tvf.getphis(), tvf.getFthetas(freq), tvf.getFphis(freq))
    tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq, vcoord='Ludwig3', projection='orthographic')

if __name__ == "__main__":
    #genDummyField()
    #testgetSphBasis()
    #testrotToFrame()
    #printsphmsh()
    #testrot3Dmat()
    if len(sys.argv) == 2:
        testplotFEKO(sys.argv[1])
    else:
        testplotFEKO(sys.argv[1], sys.argv[2])
