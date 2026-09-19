import unittest
import numpy as np
from model import objective,reflect,solve

class Tests(unittest.TestCase):
    def test_minimum(self):self.assertEqual(objective(np.zeros(9)),0)
    def test_anisotropy(self):
        a=np.zeros(9);a[0]=1;b=np.zeros(9);b[1]=1
        self.assertEqual(objective(b)/objective(a),1_000_000)
    def test_reflection(self):
        x=np.array([-110.,110.,500.,-9999.]);y=reflect(x)
        np.testing.assert_allclose(y[:2],[-90,90]);self.assertTrue(np.all(abs(y)<=100))
    def test_budget_elitism_reproducibility(self):
        c=dict(population=30,dimension=9,generations=8,tournament=3,crossover_probability=.9,mutation_probability=1/9,sigma_start=10,sigma_end=.01)
        a=solve(c,17);b=solve(c,17);r=solve(c,17,random_search=True)
        np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[3],270);self.assertEqual(a[3],r[3]);self.assertTrue(np.all(np.diff(a[2])<=0));self.assertAlmostEqual(objective(a[1]),a[0]);self.assertTrue(np.all(abs(a[1])<=100))
if __name__=='__main__':unittest.main()
