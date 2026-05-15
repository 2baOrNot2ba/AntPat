"""Read CST far-field exports stored as ASCII text tables."""

import os
import re

import numpy as np


_FREQ_RE = re.compile(r"\(f\s*=\s*([-+0-9.eE]+)\)")
_COMPONENT_RE = re.compile(r"(?:abs|phase)\(([^)]+)\)")
_FREQ_UNIT_SCALE = {
    'hz': 1.0,
    'khz': 1.0e3,
    'mhz': 1.0e6,
    'ghz': 1.0e9,
}


class CST_FF_ASCII(object):
    """Represents one CST far-field exported as an ASCII text table.

    The angular grid is inferred from the theta and phi columns in degrees.
    The vector basis is detected from the header when possible:
    Theta/Phi implies spherical polar components and U/V implies Ludwig3.
    If the header is absent or ambiguous, the default is basisType='polar'.
    Pass basisType='polar' or basisType='Ludwig3' to override detection.
    """

    def __init__(self, fn=None, freq=None, freq_unit='MHz', basisType=None):
        self.header = []
        self.theta = None
        self.phi = None
        self.abs_e = None
        self.etheta = None
        self.ephi = None
        self.axial_ratio = None
        self.F1 = None
        self.F2 = None
        self.theta_values = None
        self.phi_values = None
        self.theta_step = None
        self.phi_step = None
        self.freq = None
        self.freq_unit = freq_unit
        self.basisType = None
        if fn is not None:
            self.read(fn, freq=freq, freq_unit=freq_unit, basisType=basisType)

    def _normalize_basis_type(self, basisType):
        if basisType is None:
            return None
        lowered = basisType.lower()
        if lowered == 'polar':
            return 'polar'
        if lowered == 'ludwig3':
            return 'Ludwig3'
        raise ValueError("Unsupported basisType {}".format(basisType))

    def _infer_basis_type(self, basisType=None):
        normalized_basis = self._normalize_basis_type(basisType)
        if normalized_basis is not None:
            return normalized_basis
        if not self.header:
            return 'polar'
        normalized_header = self.header[0].lower().replace(' ', '')
        components = _COMPONENT_RE.findall(normalized_header)
        transverse_components = []
        for component in components:
            normalized_component = component.strip().lower()
            if normalized_component == 'e':
                continue
            if normalized_component not in transverse_components:
                transverse_components.append(normalized_component)
        if len(transverse_components) >= 2:
            comp0 = transverse_components[0]
            comp1 = transverse_components[1]
            if comp0 == 'theta' and comp1 == 'phi':
                return 'polar'
            if comp0 == 'u' and comp1 == 'v':
                return 'Ludwig3'
        return 'polar'

    def _infer_axis(self, values_deg, axis_name):
        axis_values = np.unique(values_deg)
        if axis_values.size == 0:
            raise ValueError("No {} samples found".format(axis_name))
        if axis_values.size == 1:
            axis_step = None
        else:
            axis_steps = np.diff(axis_values)
            axis_step = float(axis_steps[0])
            if not np.allclose(axis_steps, axis_step):
                raise ValueError(
                    "{} grid is not regularly spaced: {}"
                    .format(axis_name, axis_steps)
                )
        return axis_values, axis_step

    def _gridify(self, theta_deg, phi_deg, values, theta_values, phi_values):
        theta_count = theta_values.size
        phi_count = phi_values.size
        grid_shape = (theta_count, phi_count)
        theta_idx = np.searchsorted(theta_values, theta_deg)
        phi_idx = np.searchsorted(phi_values, phi_deg)
        grid = np.empty(grid_shape, dtype=values.dtype)
        filled = np.zeros(grid_shape, dtype=bool)
        if np.any(filled[theta_idx, phi_idx]):
            raise ValueError("CST far-field grid contains duplicate theta/phi samples")
        grid[theta_idx, phi_idx] = values
        filled[theta_idx, phi_idx] = True
        if not np.all(filled):
            raise ValueError("CST far-field grid is incomplete after theta/phi mapping")
        return grid

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

    def read(self, filename, freq=None, freq_unit='MHz', basisType=None):
        """Read a CST ASCII text far-field export.

        Parameters
        ----------
        filename : str
            CST ASCII far-field filename.
        freq : float, optional
            Frequency value to use when it should not be inferred from the
            filename.
        freq_unit : str, optional
            Unit associated with freq when it is provided explicitly.
        basisType : {'polar', 'Ludwig3'}, optional
            Override the vector basis declared by the header. If omitted, the
            basis is inferred from the header and falls back to 'polar' when
            the header is absent or ambiguous.
        """
        self.header = []
        data = self._read_data_rows(filename)
        self.basisType = self._infer_basis_type(basisType)
        theta_deg = data[:, 0]
        phi_deg = data[:, 1]
        abs_e = data[:, 2]
        abs_etheta = data[:, 3]
        phase_etheta_deg = data[:, 4]
        abs_ephi = data[:, 5]
        phase_ephi_deg = data[:, 6]
        axial_ratio = data[:, 7] if data.shape[1] > 7 else None

        theta_values, theta_step = self._infer_axis(theta_deg, 'Theta')
        phi_values, phi_step = self._infer_axis(phi_deg, 'Phi')
        theta_count = theta_values.size
        phi_count = phi_values.size
        expected_size = theta_count * phi_count
        if data.shape[0] != expected_size:
            raise ValueError(
                "CST far-field grid is incomplete: got {} samples for {}x{} grid"
                .format(data.shape[0], theta_count, phi_count)
            )

        self.theta_values = theta_values
        self.phi_values = phi_values
        self.theta_step = theta_step
        self.phi_step = phi_step

        theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing='ij')
        self.theta = theta_grid
        self.phi = phi_grid
        self.abs_e = self._gridify(theta_deg, phi_deg, abs_e,
                                   theta_values, phi_values)
        self.F1 = self._gridify(
            theta_deg, phi_deg,
            abs_etheta * np.exp(1j * np.deg2rad(phase_etheta_deg)),
            theta_values, phi_values
        )
        self.F2 = self._gridify(
            theta_deg, phi_deg,
            abs_ephi * np.exp(1j * np.deg2rad(phase_ephi_deg)),
            theta_values, phi_values
        )
        self.etheta = self.F1
        self.ephi = self.F2
        if axial_ratio is not None:
            self.axial_ratio = self._gridify(theta_deg, phi_deg, axial_ratio,
                                             theta_values, phi_values)
        else:
            self.axial_ratio = None
        self.freq = self._infer_frequency(filename, freq=freq, freq_unit=freq_unit)
        return self


def load_cst_radpat(filename, freq=None, freq_unit='MHz', basisType=None):
    """Load a CST ASCII text far-field export into a RadFarField.

    basisType defaults to 'polar' when the CST header does not identify the
    component basis explicitly.
    """
    from antpat.radfarfield import RadFarField
    from antpat.reps.sphgridfun import tvecfun

    cst_ff = CST_FF_ASCII(filename, freq=freq, freq_unit=freq_unit,
                          basisType=basisType)
    tvfd = tvecfun.TVecFields()
    tvfd._full_init(np.deg2rad(cst_ff.theta), np.deg2rad(cst_ff.phi),
                    cst_ff.F1, cst_ff.F2, R=cst_ff.freq,
                    basisType=cst_ff.basisType)
    return RadFarField(tvfd)