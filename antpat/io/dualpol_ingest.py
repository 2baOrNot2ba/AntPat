"""This module handles dual-pol files"""
import os.path
from urllib.parse import urlparse
from antpat.dualpolelem import DualPolElem
from . import filetypes as antfiles
from antpat.io.flatfile_ff import load_flatfile_radpat
from antpat.io.load_numpy import load_numpy_dpe
from antpat.reps.hamaker import convLOFARcc2DPE


def parse_path(pattern_url):
    pattern_url_parsed = urlparse(pattern_url)
    ffp_file = pattern_url_parsed.path
    basename = os.path.basename(ffp_file)
    filename, suffix = os.path.splitext(basename)
    suffix = suffix.lstrip('.')
    request = pattern_url_parsed.fragment
    return filename, suffix, request


def load_dualpol_files(pattern_url, pattern_q_url=None, filetype=None):
    """Load Dual-Pol files"""
    filename, suffix, request = parse_path(pattern_url)
    filename_q = None
    if pattern_q_url:
        filename_q, suffix_q, request_q = parse_path(pattern_q_url)
    if not filetype:
        filetype = antfiles.ffpat_suf2name(suffix)
    if filetype == antfiles.FARFIELDFORMAT_HA:
        dpe = convLOFARcc2DPE(pattern_url)
    elif filetype == antfiles.FARFIELDFORMAT_FEKO:
        dpe = DualPolElem()
        dpe.load_jones(pattern_url, pattern_q_url)
    elif filetype == antfiles.FARFIELDFORMAT_NUMPY:
        dpe = load_numpy_dpe(pattern_url)
    elif filetype == antfiles.FARFIELDFORMAT_FLAT:
        rp_p = load_flatfile_radpat(pattern_url)
        rp_q = load_flatfile_radpat(pattern_q_url)
        dpe = DualPolElem(rp_p, rp_q)
    else:
        raise RuntimeError("Unknown Dual-pol pattern file type {}"
                           .format(suffix))
    filenames = [filename, filename_q]
    return dpe, filenames, suffix

