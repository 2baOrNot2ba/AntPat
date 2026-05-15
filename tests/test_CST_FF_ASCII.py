import os
import tempfile
import unittest

import numpy as np

from antpat.io.CST_FF_ASCII import CST_FF_ASCII, load_cst_radpat


class CSTFFASCIITests(unittest.TestCase):
    def setUp(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.sample = os.path.join(root_dir, 'share', 'CST', 'farfield (f=60) [1].txt')

    def _write_cst_file(self, filepath, header, rows):
        with open(filepath, 'w') as handle:
            handle.write(header + '\n')
            handle.write('-' * max(len(header), 30) + '\n')
            for row in rows:
                handle.write(' '.join(str(value) for value in row) + '\n')

    def test_parse_cst_table(self):
        cst_ff = CST_FF_ASCII(self.sample)
        self.assertEqual(cst_ff.theta.shape, (181, 360))
        self.assertEqual(cst_ff.phi.shape, (181, 360))
        self.assertAlmostEqual(cst_ff.freq, 60.0e6)
        self.assertEqual(cst_ff.basisType, 'polar')
        self.assertAlmostEqual(cst_ff.theta_step, 1.0)
        self.assertAlmostEqual(cst_ff.phi_step, 1.0)
        self.assertAlmostEqual(cst_ff.theta[0, 0], 0.0)
        self.assertAlmostEqual(cst_ff.theta[-1, 0], 180.0)
        self.assertAlmostEqual(cst_ff.phi[0, 0], 0.0)
        self.assertAlmostEqual(cst_ff.phi[0, -1], 359.0)
        self.assertAlmostEqual(np.abs(cst_ff.etheta[0, 0]), 3.731, places=3)
        self.assertAlmostEqual(np.abs(cst_ff.ephi[0, 0]), 3.729, places=3)

    def test_parse_generic_regular_mesh(self):
        rows = [
            (10.0, 5.0, 0.5, 0.25, -30.0, 0.1, 15.0, 0.5),
            (10.0, 15.0, 1.0, 0.5, 0.0, 0.2, 90.0, 1.0),
            (0.0, 5.0, 2.0, 1.0, 180.0, 0.5, 0.0, 2.0),
            (5.0, 10.0, 3.0, 1.5, 90.0, 0.7, -90.0, 3.0),
            (5.0, 15.0, 3.5, 1.75, 60.0, 0.9, -60.0, 3.5),
            (0.0, 15.0, 4.0, 2.0, 45.0, 1.0, 180.0, 4.0),
            (0.0, 10.0, 4.5, 2.25, 10.0, 1.2, -10.0, 4.5),
            (10.0, 10.0, 5.0, 2.5, 30.0, 1.5, 45.0, 5.0),
            (5.0, 5.0, 6.0, 3.0, -45.0, 2.0, 30.0, 6.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'farfield (f=10) [1].txt')
            header = ('Theta [deg.] Phi [deg.] Abs(E)[V/m] Abs(Theta)[V/m] '
                      'Phase(Theta)[deg.] Abs(Phi)[V/m] Phase(Phi)[deg.] Ax.Ratio')
            self._write_cst_file(filepath, header, rows)

            cst_ff = CST_FF_ASCII(filepath)

        self.assertEqual(cst_ff.theta.shape, (3, 3))
        self.assertEqual(cst_ff.phi.shape, (3, 3))
        self.assertAlmostEqual(cst_ff.theta_step, 5.0)
        self.assertAlmostEqual(cst_ff.phi_step, 5.0)
        np.testing.assert_allclose(cst_ff.theta[:, 0], [0.0, 5.0, 10.0])
        np.testing.assert_allclose(cst_ff.phi[0, :], [5.0, 10.0, 15.0])
        self.assertAlmostEqual(np.abs(cst_ff.etheta[2, 2]), 0.5, places=12)
        self.assertAlmostEqual(np.angle(cst_ff.ephi[2, 2], deg=True), 90.0, places=12)

    def test_detect_ludwig3_from_header(self):
        rows = [
            (0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 90.0, 1.0),
            (0.0, 90.0, 1.0, 1.5, 0.0, 0.5, 180.0, 1.0),
            (90.0, 0.0, 1.0, 2.0, 45.0, 1.0, 0.0, 1.0),
            (90.0, 90.0, 1.0, 0.75, -45.0, 1.25, 30.0, 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'farfield (f=20) [1].txt')
            header = ('Theta [deg.] Phi [deg.] Abs(E)[V/m] Abs(U)[V/m] '
                      'Phase(U)[deg.] Abs(V)[V/m] Phase(V)[deg.] Ax.Ratio')
            self._write_cst_file(filepath, header, rows)

            cst_ff = CST_FF_ASCII(filepath)

        self.assertEqual(cst_ff.basisType, 'Ludwig3')
        self.assertAlmostEqual(np.abs(cst_ff.F1[0, 0]), 1.0)
        self.assertAlmostEqual(np.abs(cst_ff.F2[0, 0]), 2.0)

    def test_explicit_basis_type_override(self):
        rows = [
            (0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 90.0, 1.0),
            (0.0, 90.0, 1.0, 1.5, 0.0, 0.5, 180.0, 1.0),
            (90.0, 0.0, 1.0, 2.0, 45.0, 1.0, 0.0, 1.0),
            (90.0, 90.0, 1.0, 0.75, -45.0, 1.25, 30.0, 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'farfield (f=20) [1].txt')
            header = 'Theta [deg.] Phi [deg.] Abs(E)[V/m] Comp1 Comp2'
            self._write_cst_file(filepath, header, rows)

            cst_ff = CST_FF_ASCII(filepath, basisType='Ludwig3')

        self.assertEqual(cst_ff.basisType, 'Ludwig3')

    def test_ambiguous_header_defaults_to_polar(self):
        rows = [
            (0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 90.0, 1.0),
            (0.0, 90.0, 1.0, 1.5, 0.0, 0.5, 180.0, 1.0),
            (90.0, 0.0, 1.0, 2.0, 45.0, 1.0, 0.0, 1.0),
            (90.0, 90.0, 1.0, 0.75, -45.0, 1.25, 30.0, 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'farfield (f=20) [1].txt')
            header = 'Theta [deg.] Phi [deg.] Abs(E)[V/m] Comp1 Comp2'
            self._write_cst_file(filepath, header, rows)

            cst_ff = CST_FF_ASCII(filepath)

        self.assertEqual(cst_ff.basisType, 'polar')

    def test_load_tvecfields_from_cst(self):
        from antpat.reps.sphgridfun.tvecfun import TVecFields

        tvf = TVecFields()
        tvf.load_cst(self.sample)

        self.assertAlmostEqual(tvf.getRs(), 60.0e6)
        self.assertEqual(tvf.getthetas().shape, (181, 360))
        self.assertEqual(tvf.getphis().shape, (181, 360))
        self.assertAlmostEqual(np.abs(tvf.getFthetas()[0, 0]), 3.731, places=3)
        self.assertAlmostEqual(np.abs(tvf.getFphis()[0, 0]), 3.729, places=3)

    def test_load_tvecfields_from_ludwig3_cst(self):
        from antpat.reps.sphgridfun.tvecfun import TVecFields, Ludwig32sph

        rows = [
            (0.0, 90.0, 1.0, 1.5, 0.0, 0.5, 180.0, 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'farfield (f=20) [1].txt')
            header = ('Theta [deg.] Phi [deg.] Abs(E)[V/m] Abs(U)[V/m] '
                      'Phase(U)[deg.] Abs(V)[V/m] Phase(V)[deg.] Ax.Ratio')
            self._write_cst_file(filepath, header, rows)

            tvf = TVecFields()
            tvf.load_cst(filepath)

        eu = 1.5 + 0.0j
        ev = 0.5 * np.exp(1j * np.deg2rad(180.0))
        expected_eth, expected_eph = Ludwig32sph(np.deg2rad(np.array([[90.0]])),
                                                 np.array([[eu]]), np.array([[ev]]))
        self.assertAlmostEqual(tvf.getRs(), 20.0e6)
        self.assertAlmostEqual(tvf.getFthetas()[0, 0], expected_eth[0, 0])
        self.assertAlmostEqual(tvf.getFphis()[0, 0], expected_eph[0, 0])

    def test_load_radfarfield_from_cst(self):
        radpat = load_cst_radpat(self.sample)
        freqs = radpat.getfreqs()
        self.assertAlmostEqual(freqs, 60.0e6)
        theta, phi, e_th, e_ph = radpat.getFFongrid(freqs)
        self.assertEqual(theta.shape, (181, 360))
        self.assertEqual(phi.shape, (181, 360))
        self.assertAlmostEqual(np.abs(e_th[0, 0]), 3.731, places=3)
        self.assertAlmostEqual(np.abs(e_ph[0, 0]), 3.729, places=3)


if __name__ == '__main__':
    unittest.main()