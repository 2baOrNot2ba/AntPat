import numpy as np
from antpat.reps.sphgridfun import tvecfun
from antpat.dualpolelem import DualPolElem
from antpat.radfarfield import RadFarField


def load_numpy_radpat(npfile):
    """Load numpy radiation pattern"""
    with np.load(npfile) as data:
      freqs = data['freqs']
      thetamsh = data['thetamsh']
      phimsh = data['phimsh']
      E_ths = data['E_ths']
      E_phs = data['E_phs']

    tvfd = tvecfun.TVecFields()
    tvfd._full_init(thetamsh, phimsh, E_ths, E_p_phs, R=freqs)
    rp = RadFarField(tvfd)
    return rp


def load_numpy_dpe(npfile):
    """Load numpy dualpol element pattern"""
    with np.load(npfile) as data:
      freqs = data['freqs']
      thetamsh = data['thetamsh']
      phimsh = data['phimsh']
      E_p_ths = data['E_p_ths']
      E_p_phs = data['E_p_phs']
      E_q_ths = data['E_q_ths']
      E_q_phs = data['E_q_phs']
      pow_inp_p = data['pow_inp_p']
      pow_inp_q = data['pow_inp_q']

    tvfd_p = tvecfun.TVecFields()
    tvfd_p._full_init(thetamsh, phimsh, E_p_ths, E_p_phs, R=freqs)
    tvfd_q = tvecfun.TVecFields()
    tvfd_q._full_init(thetamsh, phimsh, E_q_ths, E_q_phs, R=freqs)
    rp_p = RadFarField(tvfd_p)
    rp_q = RadFarField(tvfd_q)
    dpe = DualPolElem(rp_p, rp_q)
    dpe.pow_inp_p = pow_inp_p
    dpe.pow_inp_q = pow_inp_q
    return dpe

