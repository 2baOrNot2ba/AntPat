"""Model generic, radiation far-fields."""
#TobiaC 2015-07-30

import math
import numpy
import copy
from .reps.sphgridfun import pntsonsphere, tvecfun
from .reps.vsharm.vshfield import vshField
from .reps.vsharm.hansens import Ksum


class RadFarField(object):
    """Main class for antenna far-field (FF). FF can be stored as
    tangential VSH or tangential vectors on spherical grid.
    """

    def __init__(self, tvfds):
        #tvfds="List of tangential vector field on directions
        #print tvfd_class_str
        fieldtype = type(tvfds)
        if fieldtype is vshField:
            self.rep = "VSH"
            self.vshcoefs = tvfds
        elif fieldtype is tvecfun.TVecFields:
            self.rep = "Grid"
            #tvgrid="Tangential vectors on 2-sphere grid"
            self.tvgrids = tvfds
        else:
            print("Field rep {} not found".format(fieldtype))
            exit(1)
        self.basis = None

    def getfreqs(self):
        """Get Frequencies"""
        if self.rep == "VSH":
           return self.vshcoefs.frequencies
        else:
           return self.tvgrids.getRs()

    def getFFongrid(self, freqval):
        """Get the fields over the entire sphere for a given frequency.
        For vsh fields..."""
        if self.rep == "VSH":
          THETA, PHI = pntsonsphere.sphericalGrid(100, 200)
          vshcoef = self.vshcoefs.getCoefAt(freqval)
          E_th, E_ph = Ksum(vshcoef, THETA, PHI)
        elif self.rep == "Grid":
          tvgrid = self.tvgrids.getFgridAt(freqval)
          THETA = self.tvgrids.getthetas()
          PHI = self.tvgrids.getphis()
          E_th = tvgrid[0]
          E_ph = tvgrid[1]
        return THETA, PHI, E_th, E_ph

    def getFFalong(self, freqval, theta_phi_view):
        """Get the field along some directions for a set of frequencies."""
        theta_view, phi_view = theta_phi_view
        (theta_build, phi_build) = self.view2build_coords(theta_view, phi_view)
        return self.getFFalong_build(freqval, (theta_build, phi_build))

    def getFFalong_build(self, freqval, theta_phis):
        thetas, phis = theta_phis
        if self.rep == "VSH":
            vshcoef = self.vshcoefs.getCoefAt(freqval)
            E_th, E_ph = Ksum(vshcoef, thetas, phis)
        elif self.rep == "Grid":
            E_th, E_ph = self.tvgrids.getFalong(thetas, phis, freqval)
        if self.basis is not None:
            E_th, E_ph = tvecfun.transfVecField2RotBasis(self.basis,
                                                         (thetas, phis),
                                                         (E_th, E_ph))
        E_th = numpy.squeeze(E_th)
        E_ph = numpy.squeeze(E_ph)
        return E_th, E_ph

    def view2build_coords(self, theta_view, phi_view):
        """Get the corresponding directions in the build frame."""
        if self.basis is not None:
            (theta_build, phi_build) = pntsonsphere.rotToFrame(
                                           numpy.transpose(self.basis),
                                           theta_view, phi_view)
        else:
            (theta_build, phi_build) = (theta_view, phi_view)
        return (theta_build, phi_build)

    def rotateframe(self, rotMat):
        """Rotate the frame of antenna. This is a 'passive' rotation: it
        does not rotate the field, but when evaluated in some
        direction the direction given will be rotated to the frame so
        as to appear as if it were rotated.

        The basis or rotation matrix is to be considered as acting on
        the antenna, i.e.

              view_crds=rotMat*build_crds
        """
        self.basis = rotMat

    def rotate90(self):
        """Rotate the antenna by 90 degrees around z. This is an active
        rotation on a copy of the antenna. It is intended to make an
        orthogonal instance of a linear antenna in the horizontal (xy)
        plane."""
        #Make deep copy instead of reference to list of TVecFields.
        radFFcopy = copy.deepcopy(self)
        if self.rep == "Grid":
            radFFcopy.tvgrids.rotate90z()
        elif self.rep == "VSH":
            print("VSH rotate not implimented yet.")
            exit(1)
        return radFFcopy

    def plotAntPat3D(self, freq=0.0, vcoord='sph',
                     projection='equirectangular'):
        theta_rad, phi_rad, E_th, E_ph = self.getFFongrid(freq)
        tvecfun.plotvfonsph(theta_rad, phi_rad, E_th, E_ph, freq,
                            vcoord, projection)


def makeOrtRadPat(thetaMsh, phiMsh, XEthetaMsh, XEphiMsh):
    pass
