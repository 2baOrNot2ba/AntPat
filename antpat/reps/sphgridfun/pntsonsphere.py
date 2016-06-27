"""Provides functions for points on a 2-sphere, or in other words, a set
of directions."""

import math
import numpy as np


def sphericalGrid(nrThetas=128, nrPhis=256):
    """Provides a polar angles grid on a 2-sphere. Azimuthal dimension should
    not wrap around to ensure uniqueness on sphere. The mesh has theta """
    theta = np.linspace(0.0, math.pi, nrThetas+1, endpoint=True) #Add one to
                                                                 #include endpoint
    phi = np.linspace(0.0, 2*math.pi, nrPhis, endpoint=False)
    phimsh, thetamsh = np.meshgrid(phi,theta)
    return thetamsh, phimsh


def ZenHemisphGrid(nrThetas=100, nrPhis=200):
    """Provides a polar angles grid on a 2-sphere"""
    theta = np.linspace(0., math.pi/2, nrThetas, endpoint=True)
    phi = np.linspace(0., 2*math.pi, nrPhis, endpoint=False)
    phimsh, thetamsh=np.meshgrid(phi, theta)
    return thetamsh, phimsh


def cut_theta(phicut, NrPnts=100):
    """A 1D cut along a given azimuth."""
    thetas = np.linspace(0., math.pi, NrPnts)
    phis = phicut*np.ones(thetas.shape)
    return (thetas, phis)


def cut_phi(thetacut, NrPnts=100):
    """A 1D cut along a given theta."""
    phis = np.linspace(0., 2*math.pi, NrPnts)
    thetas = thetacut*np.ones(phis.shape)
    return (thetas, phis)


def getTrack(theta0, phi0, polAng, timeAngs):
    """A 1D line of directions."""
    timeAngs_rad = math.pi/12.0*timeAngs
    thetas = np.arccos(np.cos(theta0)*np.cos(polAng)
            +np.sin(theta0)*np.sin(polAng)*np.cos(timeAngs_rad))
    phis = np.arcsin(np.sin(polAng)*np.sin(timeAngs_rad)/np.sin(theta0))
    phis = phis+phi0
    return thetas, phis


def rotToFrame(rotMat, theta_from, phi_from):
    """Rotate polar directions to a new frame given by a rotation matrix.
       The 3D rotation matrix goes from the given polar directions to the new
       polar directions.
    """
    dircos_from = sph2crtISO(theta_from, phi_from)
    dircos_to = np.matmul(rotMat, dircos_from)
    (theta_to, phi_to) = crt2sphISO(dircos_to[0], dircos_to[1], dircos_to[2])
    return (theta_to, phi_to)


def sph2crtISO(theta, phi, r=1.0):
    """Convert spherical polar angles in ISO convention to direction cosines
       vector.
    """
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return (x, y, z)


def crt2sphISO(x, y, z):
    """Convert cartesian vector to spherical polar angles in ISO convention.
    """
    phi = np.arctan2(y, x)
    theta = np.arccos(z)
    return (theta, phi)


def crt2sphHorizontal(xyz):
    """Convert cartesian vector in ENU coordinate system to spherical polar
       angles in Horizontal system. The Horizontal system has elevation angle
       from horizon and azimuth is angle from North through East.
    """
    x = xyz[0,...]
    y = xyz[1,...]
    z = xyz[2,...]
    az = np.arctan2(x, y) #The azimuth here is from North through East 
                          #but x is East and y is North, hence flip.
    el = np.arcsin(z)
    return (az, el)


def rotzmat(rotang):
    return np.array([[  np.cos(rotang),  np.sin(rotang), 0.],
                     [ -np.sin(rotang),  np.cos(rotang), 0.],
                     [              0.,              0., 1.]])


def rotxmat(rotang):
    return np.array([[1.,             0.,              0.],
                     [0.,  np.cos(rotang),  np.sin(rotang)],
                     [0., -np.sin(rotang),  np.cos(rotang)]])


def rot3Dmat(rotzang1, rotxang, rotzang0):
    """General 3D rotation matrix. Arguments follow Euler angle convention:
    rightmost arg is first rotation angle around z, then middle arg is rotation
    around x and leftmost is last rotation arouns z-axis.
    (All angles in radians)
    """
    return np.matmul(rotzmat(rotzang1), np.matmul(rotxmat(rotxang), rotzmat(rotzang0)) )

