import math
import numpy
import numpy.ma
import datetime
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from pntsonsphere import sph2crtISO, crt2sphHorizontal
from antpat.io.feko_ffe import FEKOffe, FEKOffeRequest

class TVecFields(object):
    """Provides a tangetial vector function on a spherical grid. The
    coordinates (theta,phi) should be in radians. The vector components
    can be either in polar spherical basis or in Ludwig3."""
    def __init__(self, *args):
        if len(args) > 0:
            self._full_init(*args)
    
    def _full_init(self, thetaMsh, phiMsh, F1, F2, R=None, basisType='polar'):
        self.R = R
        self.thetaMsh = thetaMsh # Assume thetaMsh is repeated columns
                                 # (unique axis=0)
        self.phiMsh = phiMsh # Assume thetaMsh is repeated rows (unique axis=1)
        if basisType == 'polar':
            self.Fthetas = F1
            self.Fphis = F2
        elif basisType == 'Ludwig3':
            # For now convert Ludwig3 components to polar spherical.
            self.Fthetas, self.Fphis = Ludwig32sph(self.phiMsh, F1, F2)
        else:
            print("Error: Unknown basisType {}".format(basisType))
            exit(1)
    
    def load_ffe(self, filename, request=None):
        ffefile = FEKOffe(filename)
        if request is None:
            if len(ffefile.Requests) == 1:
                request = ffefile.Requests.pop()
            else:
                print "File contains multiple FFs (specify one): "+','.join(ffefile.Requests)
                exit(1)
        ffereq = ffefile.Request[request]
        self.R        = numpy.array(ffereq.freqs)
        self.thetaMsh = numpy.deg2rad(ffereq.theta)
        self.phiMsh   = numpy.deg2rad(ffereq.phi)
        nrRs= len(self.R)
        self.Fthetas = numpy.zeros((nrRs, ffereq.stheta, ffereq.sphi), dtype=complex)
        self.Fphis   = numpy.zeros((nrRs, ffereq.stheta, ffereq.sphi), dtype=complex)
        # Maybe this could be done better?
        # Convert list over R of arrays over theta,phi to array over R,theta,phi
        for ridx in range(nrRs):
            self.Fthetas[ridx,:,:] = ffereq.etheta[ridx]
            self.Fphis[  ridx,:,:] = ffereq.ephi[ridx]
        # Remove redundant azimuth endpoint 2*pi
        if ffereq.phi[0,0] == 0. and ffereq.phi[0,-1] == 360.:
            self.thetaMsh = numpy.delete(self.thetaMsh, -1, 1)
            self.phiMsh = numpy.delete(self.phiMsh, -1, 1)
            self.Fthetas = numpy.delete(self.Fthetas, -1, 2)
            self.Fphis = numpy.delete(self.Fphis, -1, 2)
    
    def save_ffe(self, filename, request='FarField', source='Unknown'):
        """ """
        ffefile = FEKOffe()
        ffefile.ftype = 'Far Field'
        ffefile.fformat = '3'
        ffefile.source = source
        ffefile.date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Setup request
        ffereq = FEKOffeRequest(request)
        if self.R is not None:
            freqs = self.R
        else:
            freqs = [0.0]
        ffereq.theta   = numpy.rad2deg(self.thetaMsh)
        ffereq.phi     = numpy.rad2deg(self.phiMsh)
        coord = 'Spherical'
        stheta = ffereq.theta.shape[0]
        sphi = ffereq.phi.shape[1]
        rtype = 'Gain'
        for ridx in range(len(freqs)):
            ffereq._add_head(freqs[ridx], coord, stheta, sphi, rtype)
            if self.R is not None:
                ffereq.etheta.append(self.Fthetas[ridx,:,:].squeeze())
                ffereq.ephi.append(  self.Fphis[  ridx,:,:].squeeze())
                gtheta = numpy.abs(self.Fthetas[ridx,:,:].squeeze())**2
                gphi = numpy.abs(self.Fphis[  ridx,:,:].squeeze())**2
            else:
                ffereq.etheta.append(self.Fthetas)
                ffereq.ephi.append(  self.Fphis)
                gtheta = numpy.abs(self.Fthetas)**2
                gphi = numpy.abs(self.Fphis)**2
            gtotal = gtheta + gphi
            ffereq.gtheta.append(gtheta)
            ffereq.gphi.append(  gphi)
            ffereq.gtotal.append(gtotal)
        # Add redundant azimuth endpoint 2*pi ?

        ffefile.Requests.add(request)
        ffefile.Request[request] = ffereq
        ffefile.write(filename)
    
    def getthetas(self):
        return self.thetaMsh
    
    def getphis(self):
        return self.phiMsh
    
    def getFthetas(self, Rval=.0):
        Rind=self.getRind(Rval)
        if Rind == None:
            return self.Fthetas
        else:
            return numpy.squeeze(self.Fthetas[Rind,...])
    
    def getFphis(self, Rval=0.):
        Rind=self.getRind(Rval)
        if Rind == None:
            return self.Fphis
        else:
            return numpy.squeeze(self.Fphis[Rind,...])
    
    def getFgridAt(self, R):
        return (self.getFthetas(R), self.getFphis(R) )
    
    def getRs(self):
        return self.R
    
    def getRind(self, Rval):
        if self.R is None or type(self.R) is float:
            return None
        Rindlst = numpy.where(self.R==Rval)
        Rind = Rindlst[0][0] #For now assume unique value.
        return Rind
    
    def getFalong(self, theta_ub, phi_ub, Rval=None):
        """Get vector field for the given direction."""
        thetadomshp = theta_ub.shape
        phidomshp = phi_ub.shape
        theta_ub = theta_ub.flatten()
        phi_ub = phi_ub.flatten()
        (theta, phi) = putOnPrincBranch(theta_ub, phi_ub)
        thetaphiAxis, F_th_prdc, F_ph_prdc = periodifyRectSphGrd(self.thetaMsh,
                            self.phiMsh, self.Fthetas, self.Fphis)
        if type(self.R) is not float:
            (rM, thetaM) = numpy.meshgrid(Rval, theta, indexing='ij')
            (rM,phiM) = numpy.meshgrid(Rval, phi, indexing='ij')
            rthetaphi = numpy.zeros(rM.shape+(3,))
            rthetaphi[:,:,0] = rM
            rthetaphi[:,:,1] = thetaM
            rthetaphi[:,:,2] = phiM
            rthetaphiAxis = (self.R,)+thetaphiAxis
        else:
            rthetaphi = numpy.array([theta,phi]).T
            rthetaphiAxis = thetaphiAxis
        F_th_intrpf = RegularGridInterpolator(rthetaphiAxis, F_th_prdc)
        F_th = F_th_intrpf(rthetaphi)
        F_ph_intrpf = RegularGridInterpolator(rthetaphiAxis, F_ph_prdc)
        F_ph = F_ph_intrpf(rthetaphi)
        F_th = F_th.reshape(thetadomshp)
        F_ph = F_ph.reshape(thetadomshp)
        return F_th, F_ph
    
    def getAngRes(self):
        """Get angular resolution of mesh grid."""
        resol_th = self.thetaMsh[1,0]-self.thetaMsh[0,0]
        resol_ph = self.phiMsh[0,1]-self.phiMsh[0,0]
        return resol_th, resol_ph
      
    def sphinterp_my(self, theta, phi):
        # Currently this uses nearest value. No interpolation!
        resol_th, resol_ph  = self.getAngRes()
        ind0 = numpy.argwhere(numpy.isclose(self.thetaMsh[:,0]-theta,
                                            numpy.zeros(self.thetaMsh.shape[0]),
                                            rtol=0.0,atol=resol_th))[0][0]
        ind1 = numpy.argwhere(numpy.isclose(self.phiMsh[0,:]-phi,
                                            numpy.zeros(self.phiMsh.shape[1]),
                                            rtol=0.0,atol=resol_ph))[0][0]
        F_th = self.Fthetas[ind0,ind1]
        F_ph=  self.Fphis[ind0,ind1]
        return F_th, F_ph
    
    def rotate90z(self, sense=+1):
        self.phiMsh = self.phiMsh+sense*math.pi/2
        self.canonicalizeGrid()
    
    def canonicalizeGrid(self):
        """Put the grid into a canonical order so that azimuth goes from 0:2*pi."""
        # For now only azimuths.
        # First put all azimuthals on 0:2*pi branch:
        branchNum = numpy.floor(self.phiMsh/(2*math.pi))
        self.phiMsh = self.phiMsh-branchNum*2*math.pi
        # Assume that only columns (axis=1) have to be sorted.
        i = numpy.argsort(self.phiMsh[0,:])
        self.phiMsh = self.phiMsh[:,i]
        # thetas shouldn't need sorting on columns, but F field does:
        self.Fthetas = self.Fthetas[...,i]
        self.Fphis = self.Fphis[...,i]


def periodifyRectSphGrd(thetaMsh, phiMsh, F1, F2):
    """Create a 'periodic' function in azimuth."""
    # theta is assumed to be on [0,pi] but phi on [0,2*pi[. 
    thetaAx0 = thetaMsh[:,0].squeeze()
    phiAx0 = phiMsh[0,:].squeeze()
    phiAx = phiAx0.copy()
    phiAx = numpy.append(phiAx,phiAx0[0]+2*math.pi)
    phiAx = numpy.insert(phiAx,0,phiAx0[-1]-2*math.pi)
    F1ext = numpy.concatenate((F1[...,-1:], F1, F1[...,0:1]),axis=-1)
    F2ext=numpy.concatenate((F2[...,-1:], F2, F2[...,0:1]),axis=-1)
    return (thetaAx0, phiAx), F1ext, F2ext


def putOnPrincBranch(theta,phi):
    branchNum = numpy.floor(phi/(2*math.pi))
    phi_pb = phi-branchNum*2*math.pi
    theta = numpy.abs(theta)
    branchNum = numpy.round(theta/(2*math.pi))
    theta_pb = numpy.abs(theta-branchNum*2*math.pi)
    return (theta_pb, phi_pb)


def transfVecField2RotBasis(basisto, thetas_phis_build, F_th_ph):
    """This is essentially a parallactic rotation of the transverse field."""
    thetas_build, phis_build = thetas_phis_build
    F_th, F_ph = F_th_ph
    xyz = numpy.asarray(sph2crtISO(thetas_build, phis_build))
    xyzto = numpy.matmul(basisto, xyz)
    sphcrtMat = getSph2CartTransfMatT(xyz, ISO=True)
    sphcrtMatto = getSph2CartTransfMatT(xyzto, ISO=True)
    sphcrtMatfrom_to = numpy.matmul(numpy.transpose(basisto), sphcrtMatto)
    parRot = numpy.matmul(numpy.swapaxes(sphcrtMat[:,:,1:], 1, 2),
                        sphcrtMatfrom_to[:,:,1:])
    F_thph = numpy.rollaxis(numpy.array([F_th, F_ph]), 0, F_th.ndim+1
                           )[...,numpy.newaxis]
    F_thph_to = numpy.rollaxis(numpy.matmul(parRot, F_thph).squeeze(), -1, 0)
    return F_thph_to


def getSph2CartTransfMat(rvm, ISO=False):
    """Compute the transformation matrix from a spherical basis to a Cartesian
    basis at the field point given by the input 'r'. If input 'r' is an array
    with dim>1 then the last dimension holds the r vector components.
    The output 'transf_sph2cart' is defined such that:
    
    [[v_x], [v_y], [v_z]]=transf_sph2cart*matrix([[v_r], [v_phi], [v_theta]]).
    for non-ISO case.
    
    Returns transf_sph2cart[si,ci,bi] where si,ci,bi are the sample index,
    component index, and basis index resp.
    The indices bi=0,1,2 map to r,phi,theta for non-ISO otherwise they map to
    r,theta,phi resp., while ci=0,1,2 map to xhat, yhat, zhat resp."""
    nrOfrv = rvm.shape[0]
    rabs = numpy.sqrt(rvm[:,0]**2+rvm[:,1]**2+rvm[:,2]**2)
    rvmnrm = rvm/rabs[:,numpy.newaxis]
    xu = rvmnrm[:,0]
    yu = rvmnrm[:,1]
    zu = rvmnrm[:,2]
    rb = numpy.array([xu, yu, zu])
    angnrm = 1.0/numpy.sqrt(xu*xu+yu*yu)
    phib = angnrm*numpy.array([yu, -xu, numpy.zeros(nrOfrv)])
    thetab = angnrm*numpy.array([xu*zu, yu*zu, -(xu*xu+yu*yu)])
    if ISO:
        transf_sph2cart = numpy.array([rb, thetab, phib])
    else:
        transf_sph2cart = numpy.array([rb, phib, thetab])
    # Transpose the result to get output as stack of transform matrices:
    transf_sph2cart = numpy.transpose(transf_sph2cart, (2,1,0))
    
    return transf_sph2cart


def getSph2CartTransfMatT(rvm, ISO=False):
    """Analogous to previous but with input transposed. """
    shOfrv = rvm.shape[1:]
    dmOfrv = rvm.ndim-1
    rabs = numpy.sqrt(rvm[0]**2+rvm[1]**2+rvm[2]**2)
    rvmnrm = rvm/rabs
    xu = rvmnrm[0]
    yu = rvmnrm[1]
    zu = rvmnrm[2]
    rb = numpy.array([xu, yu, zu])
    nps=rb[2,...]==1.0
    rho = numpy.sqrt(xu*xu+yu*yu)
    npole = numpy.where(rho==0.)
    rho[npole]=numpy.finfo(float).tiny
    angnrm = 1.0/rho
    phib = angnrm*numpy.array([yu, -xu, numpy.zeros(shOfrv)])
    thetab = angnrm*numpy.array([xu*zu, yu*zu, -(xu*xu+yu*yu)])
    if len(npole[0])>0:
        phib[:,nps] = numpy.array([0, 1, 0])[:,None]
        thetab[:,nps] = numpy.array([1, 0, 0])[:,None]
    # CHECK signs of basis!
    if ISO:
        transf_sph2cart = numpy.array([rb, thetab, phib])
    else:
        transf_sph2cart = numpy.array([rb, -phib, thetab])
    # Transpose the result to get output as stack of transform matrices:
    transf_sph2cart = numpy.rollaxis(transf_sph2cart, 0, dmOfrv+2)
    transf_sph2cart = numpy.rollaxis(transf_sph2cart, 0, dmOfrv+2-1)
    return transf_sph2cart


def plotAntPat2D(angle_rad, F_th, F_ph, freq=0.5):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    angle = numpy.rad2deg(angle_rad)
    ax1.plot(angle, numpy.abs(F_th), label="F_th")
    ax1.plot(angle, numpy.abs(F_ph), label="F_ph")
    ax2 = fig.add_subplot(212)
    ax2.plot(angle, numpy.rad2deg(F_th))
    ax2.plot(angle, numpy.rad2deg(F_ph))
    plt.show()


def plotFEKO(filename, request=None, freq_req=None):
    """Convenience function that reads in FEKO FFE files - using load_ffe() - and
    plots it - using plotvfonsph()."""
    tvf = TVecFields()
    tvf.load_ffe(filename, request)
    freqs = tvf.getRs()
    #frqIdx = numpy.where(numpy.isclose(freqs,freq,atol=190e3))[0][0]
    if freq_req is None:
        print("")
        print("No user specified frequency (will choose first in list)")
        print("List of frequencies (in Hz):")
        print(", ".join([str(f) for f in freqs]))
        print("")
        frqIdx = 0
    else:
        frqIdx = numpy.interp(freq_req, freqs, range(len(freqs)))
    freq = freqs[frqIdx]
    print("Frequency={}".format(freq))
    (THETA, PHI, E_th, E_ph) = (tvf.getthetas(), tvf.getphis(), tvf.getFthetas(freq), tvf.getFphis(freq))
    plotvfonsph(THETA, PHI, E_th, E_ph, freq, vcoord='Ludwig3', projection='orthographic')


#TobiaC (2013-06-17)
def projectdomain(theta_rad, phi_rad, F_th, F_ph, projection):
    """Convert spherical coordinates into various projections."""
    projections = ['orthographic', 'azimuthal-equidistant', 'equirectangular']
    if projection == 'orthographic':
        #Fix check for theta>pi/2
        #Plot hemisphere theta<pi/2
        UHmask = theta_rad>math.pi/2
        F_th = numpy.ma.array(F_th, mask=UHmask)
        F_ph = numpy.ma.array(F_ph, mask=UHmask)
        x = numpy.sin(theta_rad)*numpy.cos(phi_rad)
        y = numpy.sin(theta_rad)*numpy.sin(phi_rad)
        xyNames = ('l','m')
        nom_xticks=None
    elif projection == 'azimuthal-equidistant':
        # 2D polar to cartesian conversion
        # (put in offset)
        x = theta_rad*numpy.cos(phi_rad)
        y = theta_rad*numpy.sin(phi_rad)
        xyNames = ('theta*cos(phi)','theta*sin(phi)')
        nom_xticks=None
    elif projection == 'equirectangular':
        y = theta_rad
        x = phi_rad
        xyNames = ('phi','theta')
        nom_xticks=None #[0,45,90,135,180,225,270,315,360]
    else:
        print("Supported projections are: {}".format(', '.join(projections)))
        raise ValueError("Unknown map projection: {}".format(projection))
    return x, y, xyNames, nom_xticks, F_th, F_ph


def lin2circ(vx, vy, isign=1):
    """Convert 2-vector from linear basis to circular basis. Output order L, R.
    isign argument chooses sign of imaginary unit in phase convention. (See Hamaker1996_III)"""
    vl = (vx-isign*1j*vy)/math.sqrt(2)
    vr = (vx+isign*1j*vy)/math.sqrt(2)
    return vl, vr


def circ2lin(vl,vr, isign=1):
    """Convert 2-vector from circular basis to linear basis. Input order L, R.
    isign argument chooses sign of imaginary unit in phase convention. (See Hamaker1996_III)"""
    vx =          (vl+vr)/math.sqrt(2)
    vy = isign*1j*(vl-vr)/math.sqrt(2)
    return vx, vy


def vcoordconvert(F1, F2, phi_rad, vcoordlist):
    """Convert transverse vector components of field."""
    vcoords = ['Ludwig3', 'sph', 'circ', 'lin']
    compname =['F_', 'F_']
    for vcoord in vcoordlist:
        if vcoord == 'Ludwig3':
            F1p, F2p = sph2Ludwig3(phi_rad, F1, F2)
            compsuffix = ['u', 'v']
        elif vcoord == 'sph':
            F1p, F2p = F1, F2
            compsuffix = ['theta', 'phi']
        elif vcoord == 'circ':
            F1p, F2p = lin2circ(F1, F2)
            compsuffix = ['L', 'R']
        elif vcoord == 'lin':
            F1p, F2p = circ2lin(F1, F2)
            compsuffix = ['X', 'Y']
        else:
            raise ValueError("Unknown vector coord sys")
        compname = [compname[0]+compsuffix[0], compname[1]+compsuffix[1]]
        F1, F2 = F1p, F2p
    return F1, F2, compname


def cmplx2realrep(F_c, cmplx_rep):
    """Complex to real representation"""
    if cmplx_rep=='ReIm':
        cmpopname_r0, cmpopname_r1= 'Re', 'Im'
        F_r0, F_r1 = numpy.real(F_c), numpy.imag(F_c)
    elif cmplx_rep=='AbsAng':
        cmpopname_r0, cmpopname_r1= 'Abs', 'Ang'
        F_r0, F_r1 = numpy.absolute(F_c), numpy.rad2deg(numpy.angle(F_c))
    else:
        raise ValueError("Complex representation not known")
    return (F_r0, F_r1), (cmpopname_r0, cmpopname_r1)


# This function should be recast as refering to radial component instead of freq.
def plotvfonsph(theta_rad, phi_rad, F_th, F_ph, freq=0.0,
                vcoordlist=['sph'], projection='orthographic', cmplx_rep='AbsAng',
                vfname='Unknown'):
    """Plot transverse vector field on sphere. Different projections are
    supported as are different bases and complex value representations."""
    x, y, xyNames, nom_xticks, F_th, F_ph = projectdomain(theta_rad, phi_rad,
                                                         F_th, F_ph, projection)
    F0_c, F1_c, compNames =  vcoordconvert(F_th, F_ph, phi_rad, vcoordlist=vcoordlist)
    F0_2r, cmplxop0 = cmplx2realrep(F0_c, cmplx_rep)
    F1_2r, cmplxop1 = cmplx2realrep(F1_c, cmplx_rep)
    if projection == 'orthographic' or projection == 'azimuthal-equidistant':
        x = numpy.rad2deg(x)
        y = numpy.rad2deg(y)
        xyNames = [xyNames[0]+' [deg.]', xyNames[1]+' [deg.]']
    fig = plt.figure()
    fig.suptitle(vfname+' @ '+str(freq/1e6)+' MHz'+', '
                 +'projection: '+projection)
    
    def plotcomp(vcmpi, cpi, zcomp, cmplxop, xyNames, nom_xticks):
        if cmplxop[cpi] == 'Ang':
            cmap = plt.get_cmap('hsv')
        else:
            cmap = plt.get_cmap('viridis')
        plt.pcolormesh(x, y, zcomp[cpi], cmap=cmap)
        if nom_xticks is not None: plt.xticks(nom_xticks)
        # FIX next line
        ax.set_title(cmplxop[cpi]+'('+compNames[vcmpi]+')')
        plt.xlabel(xyNames[0])
        plt.ylabel(xyNames[1])
        plt.grid()
        plt.colorbar()
        if projection is not 'orthographic':
            ax.invert_yaxis()
    
    ax = plt.subplot(221,polar=False)
    plotcomp(0, 0, F0_2r, cmplxop0, xyNames, nom_xticks)
    ax = plt.subplot(222,polar=False)
    plotcomp(0, 1, F0_2r, cmplxop0, xyNames, nom_xticks)
    ax = plt.subplot(223,polar=False)
    plotcomp(1, 0, F1_2r, cmplxop1, xyNames, nom_xticks)
    ax = plt.subplot(224,polar=False)
    plotcomp(1, 1, F1_2r, cmplxop1, xyNames, nom_xticks)
    
    plt.show()


def plotvfonsph3D(theta_rad, phi_rad, E_th, E_ph, freq=0.0,
                     vcoord='sph', projection='equirectangular'):
    PLOT3DTYPE = "quiver"
    (x, y, z) = sph2crtISO(theta_rad, phi_rad)
    from mayavi import mlab
    
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    if PLOT3DTYPE == "MESH_RADIAL" :
        r_Et = numpy.abs(E_th)
        r_Etmx = numpy.amax(r_Et)
        mlab.mesh(r_Et*(x)-1*r_Etmx, r_Et*y, r_Et*z, scalars=r_Et)
        r_Ep = numpy.abs(E_ph)
        r_Epmx = numpy.amax(r_Ep)
        mlab.mesh(r_Ep*(x)+1*r_Epmx , r_Ep*y, r_Ep*z, scalars=r_Ep)
    elif PLOT3DTYPE == "quiver":
        ##Implement quiver plot
        s2cmat = getSph2CartTransfMatT(numpy.array([x,y,z]))
        E_r = numpy.zeros(E_th.shape)
        E_fldsph = numpy.rollaxis(numpy.array([E_r, E_ph, E_th]), 0, 3)[...,numpy.newaxis]
        E_fldcrt = numpy.rollaxis(numpy.matmul(s2cmat, E_fldsph).squeeze(), 2, 0)
        #print E_fldcrt.shape
        mlab.quiver3d(x+1.5, y, z,
                      numpy.real(E_fldcrt[0]),
                      numpy.real(E_fldcrt[1]),
                      numpy.real(E_fldcrt[2]))
        mlab.quiver3d(x-1.5, y, z,
                      numpy.imag(E_fldcrt[0]),
                      numpy.imag(E_fldcrt[1]),
                      numpy.imag(E_fldcrt[2]))              
    mlab.show()


def sph2Ludwig3(azl, EsTh, EsPh):
    """Input: an array of theta components and an array of phi components.
    Output: an array of Ludwig u components and array Ludwig v.
    Ref Ludwig1973a."""
    EsU = EsTh*numpy.sin(azl)+EsPh*numpy.cos(azl)
    EsV = EsTh*numpy.cos(azl)-EsPh*numpy.sin(azl)
    return EsU, EsV


def Ludwig32sph(azl, EsU, EsV):
    EsTh = EsU*numpy.sin(azl)+EsV*numpy.cos(azl)
    EsPh = EsU*numpy.cos(azl)-EsV*numpy.sin(azl)
    return EsTh, EsPh
