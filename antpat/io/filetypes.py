FEKOsuffix = 'ffe'
GRASPsuffix = 'sph'  # aka SWE file
NECsuffix = 'out'
HamArtsuffix = 'cc'
Numpysuffix = 'npz'

FARFIELDFORMAT_FEKO = 'ffe'
FARFIELDFORMAT_GRASP = 'sph'
FARFIELDFORMAT_NEC = 'nec'
FARFIELDFORMAT_HA = 'ha'
FARFIELDFORMAT_NUMPY = 'np'
FARFIELDFORMAT_FLAT = 'flatfile'

def ffpat_suf2name(suffix):
    """Farfield pattern file suffix to name

    Parameters
    ----------
    suffix : str
        Farfield pattern's file suffix

    Returns
    -------
    filetype : str
        Farfield pattern's filetype name.
        Currently:
            * 'ffe': FEKO's ffe file.
            * 'sph': GRASP's SWE file.
            * 'nec': NEC's out file.
            * 'ha': Hamaker-Arts cc file.
            * 'np': Numpy file.
            * '': Comma separated values file.
     """
    if suffix == FEKOsuffix:
        filetype = FARFIELDFORMAT_FEKO
    elif suffix == GRASPsuffix:
        filetype = FARFIELDFORMAT_GRASP
    elif suffix == NECsuffix:
        filetype = FARFIELDFORMAT_NEC
    elif suffix == HamArtsuffix:
        filetype = FARFIELDFORMAT_HA
    elif suffix == Numpysuffix:
        filetype = FARFIELDFORMAT_NUMPY
    else:
        filetype = FARFIELDFORMAT_FLAT
    return filetype

