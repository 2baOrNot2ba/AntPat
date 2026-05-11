"""Read CST far-field exports stored as ASCII text tables."""

import os
import re

import numpy as np


_FREQ_RE = re.compile(r"\(f\s*=\s*([-+0-9.eE]+)\)")
_FREQ_UNIT_SCALE = {
    'hz': 1.0,
    'khz': 1.0e3,
    'mhz': 1.0e6,
    'ghz': 1.0e9,
}


class CST_FF_ASCII(object):
    """Represents one CST far-field exported as an ASCII text table."""

    def __init__(self, fn=None, freq=None, freq_unit='MHz'):
        self.header = []
        self.theta = None
        self.phi = None
        self.abs_e = None
        self.etheta = None
        self.ephi = None
        self.axial_ratio = None
        self.freq = None
        self.freq_unit = freq_unit
        if fn is not None:
            self.read(fn, freq=freq, freq_unit=freq_unit)

    def _infer_frequency(self, filename, freq=None, freq_unit='MHz'):
        if freq is None:
            match = _FREQ_RE.search(os.path.basename(filename))
            if match is None:
                raise ValueError(
                    "Could not infer frequency from filename {}; pass freq explicitly"
                    .format(filename)
                )
            freq = float(match.group(1))
        scale = _FREQ_UNIT_SCALE.get(freq_unit.lower())
        if scale is None:
            raise ValueError("Unsupported frequency unit {}".format(freq_unit))
        self.freq_unit = freq_unit
        return float(freq) * scale

    def _read_data_rows(self, filename):
        data_rows = []
        in_data = False
        with open(filename, 'r') as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if not in_data:
                    self.header.append(stripped)
                    if set(stripped) == {'-'}:
                        in_data = True
                    continue
                data_rows.append([float(value) for value in stripped.split()])
        if not data_rows:
            raise ValueError("No CST far-field samples found in {}".format(filename))
        data = np.asarray(data_rows, dtype=float)
        if data.shape[1] < 7:
            raise ValueError(
                "Expected at least 7 columns in CST far-field table, got {}"
                .format(data.shape[1])
            )
        return data

    def read(self, filename, freq=None, freq_unit='MHz'):
        """Read a CST ASCII text far-field export."""
        self.header = []
        data = self._read_data_rows(filename)
        theta_deg = data[:, 0]
        phi_deg = data[:, 1]
        abs_e = data[:, 2]
        abs_etheta = data[:, 3]
        phase_etheta_deg = data[:, 4]
        abs_ephi = data[:, 5]
        phase_ephi_deg = data[:, 6]
        axial_ratio = data[:, 7] if data.shape[1] > 7 else None

        theta_count = np.unique(theta_deg).size
        phi_count = np.unique(phi_deg).size
        expected_size = theta_count * phi_count
        if data.shape[0] != expected_size:
            raise ValueError(
                "CST far-field grid is incomplete: got {} samples for {}x{} grid"
                .format(data.shape[0], theta_count, phi_count)
            )

        grid_shape = (theta_count, phi_count)
        self.theta = theta_deg.reshape(grid_shape, order='F')
        self.phi = phi_deg.reshape(grid_shape, order='F')
        self.abs_e = abs_e.reshape(grid_shape, order='F')
        self.etheta = (
            abs_etheta * np.exp(1j * np.deg2rad(phase_etheta_deg))
        ).reshape(grid_shape, order='F')
        self.ephi = (
            abs_ephi * np.exp(1j * np.deg2rad(phase_ephi_deg))
        ).reshape(grid_shape, order='F')
        if axial_ratio is not None:
            self.axial_ratio = axial_ratio.reshape(grid_shape, order='F')
        else:
            self.axial_ratio = None
        self.freq = self._infer_frequency(filename, freq=freq, freq_unit=freq_unit)
        return self


def load_cst_radpat(filename, freq=None, freq_unit='MHz'):
    """Load a CST ASCII text far-field export into a RadFarField."""
    from antpat.radfarfield import RadFarField
    from antpat.reps.sphgridfun import tvecfun

    cst_ff = CST_FF_ASCII(filename, freq=freq, freq_unit=freq_unit)
    tvfd = tvecfun.TVecFields()
    tvfd._full_init(np.deg2rad(cst_ff.theta), np.deg2rad(cst_ff.phi),
                    cst_ff.etheta, cst_ff.ephi, R=cst_ff.freq)
    return RadFarField(tvfd)