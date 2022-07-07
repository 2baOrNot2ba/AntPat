"""This module handles dual-pol files"""
import os.path
from urllib.parse import urlparse
from antpat.dualpolelem import DualPolElem
from . import filetypes as antfiles


def parse_path(pattern_url):
    pattern_url_parsed = urlparse(pattern_url)
    ffp_file = pattern_url_parsed.path
    basename = os.path.basename(ffp_file)
    filename, suffix = os.path.splitext(basename)
    suffix = suffix.lstrip('.')
    request = pattern_url_parsed.fragment
    return filename, suffix, request


def load_dualpol_files(pattern_url, pattern_q_url=None):
    """Load Dual-Pol files"""
    filename, suffix, request = parse_path(pattern_url)
    if pattern_q_url:
        filename_q, suffix_q, request_q = parse_path(pattern_q_url)
    if suffix == antfiles.HamArtsuffix:
        dpe = convLOFARcc2DPE(args.filename)
    elif suffix == antfiles.FEKOsuffix:
        dpe = DualPolElem()
        dpe.load_jones(pattern_url, pattern_q_url)
    else:
        raise RuntimeError("Unknown Dual-pol pattern file type {}"
                           .format(suffix))
    filenames = [filename, filename_q]
    return dpe, filenames, suffix
