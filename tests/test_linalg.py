import random
import unittest

from _util import la, rel_diff


class TestLinalg(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1234)

    def test_sym_eig_reconstructs_and_orthonormal(self):
        a = la.random_normal(7, 7, self.rng)
        s = la.symmetrize(la.add(a, la.transpose(a)))
        w, v = la.sym_eig(s)
        self.assertEqual(w, sorted(w, reverse=True))
        rec = la.matmul(la.matmul(v, la.diag(w)), la.transpose(v))
        self.assertLess(rel_diff(rec, s), 1e-12)
        self.assertLess(rel_diff(la.matmul(la.transpose(v), v), la.eye(7)), 1e-12)

    def test_svd_tall_wide_and_rank_deficient(self):
        for m, n in [(6, 4), (4, 6), (5, 5)]:
            a = la.random_normal(m, n, self.rng)
            u, s, v = la.svd(a)
            p = min(m, n)
            self.assertEqual(la.shape(u), (m, p))
            self.assertEqual(la.shape(v), (n, p))
            rec = la.matmul(la.matmul(u, la.diag(s)), la.transpose(v))
            self.assertLess(rel_diff(rec, a), 1e-12)
            self.assertLess(rel_diff(la.matmul(la.transpose(u), u), la.eye(p)), 1e-12)
            self.assertLess(rel_diff(la.matmul(la.transpose(v), v), la.eye(p)), 1e-12)
            self.assertEqual(s, sorted(s, reverse=True))
        # rank-2 matrix of shape 6x5: U must still be orthonormal (completed)
        b = la.matmul(la.random_normal(6, 2, self.rng), la.random_normal(2, 5, self.rng))
        u, s, v = la.svd(b)
        self.assertLess(s[2] / s[0], 1e-12)
        self.assertLess(rel_diff(la.matmul(la.transpose(u), u), la.eye(5)), 1e-10)
        self.assertEqual(la.numerical_rank(b), 2)

    def test_eckart_young_error(self):
        a = la.random_normal(6, 5, self.rng)
        s = la.singular_values(a)
        for k in range(0, 6):
            ak = la.svd_truncate(a, k)
            err = la.frob2(la.sub(a, ak))
            self.assertAlmostEqual(err, sum(x * x for x in s[k:]), places=10)
            self.assertLessEqual(la.numerical_rank(ak), k)

    def test_solve_and_inv(self):
        a = la.add(la.random_normal(5, 5, self.rng), la.scale(3.0, la.eye(5)))
        b = la.random_normal(5, 3, self.rng)
        x = la.solve(a, b)
        self.assertLess(rel_diff(la.matmul(a, x), b), 1e-12)
        self.assertLess(rel_diff(la.matmul(a, la.inv(a)), la.eye(5)), 1e-12)
        with self.assertRaises(la.LinAlgError):
            la.solve([[1.0, 2.0], [2.0, 4.0]], [[1.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
