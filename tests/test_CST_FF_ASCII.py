import os
import unittest

import numpy as np

from antpat.io.CST_FF_ASCII import CST_FF_ASCII, load_cst_radpat


class CSTFFASCIITests(unittest.TestCase):
    def setUp(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.sample = os.path.join(root_dir, 'share', 'CST', 'farfield (f=60) [1].txt')

    def test_parse_cst_table(self):
        cst_ff = CST_FF_ASCII(self.sample)
        self.assertEqual(cst_ff.theta.shape, (181, 360))
        self.assertEqual(cst_ff.phi.shape, (181, 360))
        self.assertAlmostEqual(cst_ff.freq, 60.0e6)
        self.assertAlmostEqual(cst_ff.theta[0, 0], 0.0)
        self.assertAlmostEqual(cst_ff.theta[-1, 0], 180.0)
        self.assertAlmostEqual(cst_ff.phi[0, 0], 0.0)
        self.assertAlmostEqual(cst_ff.phi[0, -1], 359.0)
        self.assertAlmostEqual(np.abs(cst_ff.etheta[0, 0]), 3.731, places=3)
        self.assertAlmostEqual(np.abs(cst_ff.ephi[0, 0]), 3.729, places=3)

    def test_load_tvecfields_from_cst(self):
        from antpat.reps.sphgridfun.tvecfun import TVecFields

        tvf = TVecFields()
        tvf.load_cst(self.sample)

        self.assertAlmostEqual(tvf.getRs(), 60.0e6)
        self.assertEqual(tvf.getthetas().shape, (181, 360))
        self.assertEqual(tvf.getphis().shape, (181, 360))
        self.assertAlmostEqual(np.abs(tvf.getFthetas()[0, 0]), 3.731, places=3)
        self.assertAlmostEqual(np.abs(tvf.getFphis()[0, 0]), 3.729, places=3)

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