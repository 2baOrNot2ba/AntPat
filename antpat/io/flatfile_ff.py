import numpy
from antpat.reps.sphgridfun import tvecfun
from antpat.radfarfield import RadFarField


def _load_flatfile_ff(filename):
    """Load a farfield from a flat-file

    Parameters
    ----------
    filename : str
        Name of flat-file with farfield pattern

    Returns
    -------
    freqs : array of real floats
        Frequencies
    thetas : array of real floats
        Polar angle theta in radians
    phis : array of real floats
        Azimuthal angle in radians
    E_th : array of complex floats
        Theta component of electric field.
    E_ph : array of complex floats
        Phi component of electric field.
    pow_input : array of real floats
        Input power.
    pow_rad :
        Output power.
    """
    out = numpy.loadtxt(filename)
    (freqs, thetas, phis, pow_input, pow_rad, Ephi_re, Ephi_im, Eth_re, Eth_im) \
     = (out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4],  out[:, 5],
        out[:, 6], out[:, 7],  out[:, 8])
    thetas = numpy.deg2rad(thetas)
    phis = numpy.deg2rad(phis)
    E_ph = Ephi_re + 1j*Ephi_im
    E_th =  Eth_re + 1j*Eth_im

    return freqs, thetas, phis, E_th, E_ph, pow_input, pow_rad


def _unravel_1x2D_old(c0_flat, c1_flat, c2_flat, f1_flat, f2_flat, major_c1=True):
    """Unravel a three coordinate flatten array

    If major_c1 then 2d output is put into c1 major order otherwise c2 major.
    """
    c0 = numpy.unique(c0_flat)  # Unique freqs
    c1 = numpy.unique(c1_flat)
    c2 = numpy.unique(c2_flat)
    # Assume that data is c0 major ordered in 1D
    # Determine which 2D coordinate is major, c1 or c2;
    # take 1st column of c1msh and make elements unique.
    majorslice_uniq_c1 =  numpy.unique(c1_flat[0:len(c2)])
    # and determine if we need to transpose to change major order
    do_transp = False
    if len(majorslice_uniq_c1) == 1:
        # c1 is major coord
        flat2d_shape = len(c1), len(c2)
        if not major_c1:
            do_transp = True
    else:
        # c2 is major coord
        flat2d_shape = len(c2), len(c1)
        if major_c1:
            do_transp = True
    # Unravel data
    # unravel 2D coords
    c1msh = c1_flat.reshape(flat2d_shape)
    c2msh = c2_flat.reshape(flat2d_shape)
    # unravel fields
    f1 = f1_flat.reshape((len(c0),)+flat2d_shape)
    f2 = f2_flat.reshape((len(c0),)+flat2d_shape)
    if do_transp:
        # Transpose 2D axes 1,2 while axis 0 is untouched
        c1msh = c1msh.transpose()
        c2msh = c2msh.transpose()
        f1 = f1.transpose((0,2,1))
        f2 = f2.transpose((0,2,1))
    if len(c0) == 1:
        # Squeeze singleton c0 coord
        sqz_sngltn = True
        if sqz_sngltn:
            c0 = float(c0[0])
            f1 = f1.squeeze()
            f2 = f2.squeeze()
    return c0, c1msh, c2msh, f1, f2


def _unravel_1x2D(c0_flat, c1_flat, c2_flat, f1_flat, f2_flat, major_c1=True):
    """Unravel a three coordinate flatten array

    If major_c1 then 2d output is put into c1 major order otherwise c2 major.
    """
    c0 = numpy.unique(c0_flat)  # Unique freqs
    c1 = numpy.unique(c1_flat)
    c2 = numpy.unique(c2_flat)
    # Assume that data is c0 major ordered in 1D
    # Determine which 2D coordinate is major, c1 or c2;
    # take 1st column of c1msh and make elements unique.
    majorslice_uniq_c1 =  numpy.unique(c1_flat[0:len(c2)])
    # and determine if we need to transpose to change major order
    do_transp = False
    if len(majorslice_uniq_c1) == 1:
        # c1 is major coord
        flat2d_shape = len(c1), len(c2)
        if not major_c1:
            do_transp = True
    else:
        # c2 is major coord
        flat2d_shape = len(c2), len(c1)
        if major_c1:
            do_transp = True
    ord='C'
    if do_transp:
        flat2d_shape = tuple(reversed(flat2d_shape))
        ord = 'F'
    # Unravel data
    # unravel 2D coords
    c1msh = c1_flat.reshape(flat2d_shape, order=ord)
    c2msh = c2_flat.reshape(flat2d_shape, order=ord)
    # unravel fields
    f1 = f1_flat.reshape((len(c0),)+flat2d_shape, order=ord)
    f2 = f2_flat.reshape((len(c0),)+flat2d_shape, order=ord)
    if len(c0) == 1:
        # Squeeze singleton c0 coord
        sqz_sngltn = True
        if sqz_sngltn:
            c0 = float(c0[0])
            f1 = f1.squeeze()
            f2 = f2.squeeze()
    return c0, c1msh, c2msh, f1, f2


def load_flatfile_radpat(filename):
    """Load a radpat object from a flat-file

    Parameters
    ----------
    filename : str
        Name of flat-file with farfield pattern

    Returns
    -------
    radpat : RadFarField
        Radiation far-field object.
    """
    (freq_flat, theta_flat, phi_flat, E_th_flat, E_ph_flat, pow_input_flat,
     pow_rad_flat) = _load_flatfile_ff(filename)
    freqs, thetamsh, phimsh, E_ths, E_phs \
        = _unravel_1x2D(freq_flat, theta_flat, phi_flat, E_th_flat, E_ph_flat)
    pow_input = numpy.unique(pow_input_flat)
    pow_rad = numpy.unique(pow_rad_flat)
    atvfd = tvecfun.TVecFields()
    atvfd._full_init(thetamsh, phimsh, E_ths, E_phs, R=freqs)
    antFF = RadFarField(atvfd)
    antFF.pow_inp = pow_input
    antFF.pow_rad = pow_rad
    return antFF

if __name__ == "__main__":
    import sys
    rp = load_flatfile_radpat(sys.argv[1])
    thetas=numpy.radians(numpy.array([0.5]))
    phis=numpy.radians(numpy.array([0.0]))
    print(rp.getFFalong(rp.getfreqs(), (thetas, phis)))
    
