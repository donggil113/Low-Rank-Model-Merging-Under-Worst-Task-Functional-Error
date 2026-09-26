"""Property checks of the draft's perturbation inequalities (Lemma 1 and Lemma 3)."""

import random
import unittest

from _util import la

from lowrank_merge.run_bound_check import e_val, op_norm_sym


def rand_psd(rng, q, n):
    x = la.random_normal(q, n, rng)
    return la.scale(1.0 / n, la.gram_rows(x))


class TestLemmas(unittest.TestCase):
    def test_op_norm_matches_largest_abs_eigenvalue(self):
        a = [[2.0, 0.0], [0.0, -3.0]]
        self.assertAlmostEqual(op_norm_sym(a), 3.0, places=12)

    def test_lemma1_and_lemma3_hold_on_random_instances(self):
        rng = random.Random(0)
        for _ in range(200):
            p, q = rng.choice([(3, 4), (5, 5)])
            d = la.random_normal(p, q, rng)
            w = la.random_normal(p, q, rng)
            s = rand_psd(rng, q, 50)
            sh = rand_psd(rng, q, rng.choice([3, 8, 20]))
            dop = op_norm_sym(la.sub(s, sh))
            d_true = la.inner(la.matmul(d, s), d)
            d_hat = la.inner(la.matmul(d, sh), d)
            # Lemma 1 (fixed normaliser)
            lhs1 = abs(e_val(w, d, s, d_true) - e_val(w, d, sh, d_true))
            rhs1 = dop * la.frob2(la.sub(w, d)) / d_true
            self.assertLessEqual(lhs1, rhs1 * (1 + 1e-12) + 1e-15)
            # Lemma 3 (estimated normaliser)
            true_e = e_val(w, d, s, d_true)
            lhs3 = abs(true_e - e_val(w, d, sh, d_hat))
            rhs3 = dop / d_hat * (la.frob2(la.sub(w, d)) + true_e * la.frob2(d))
            self.assertLessEqual(lhs3, rhs3 * (1 + 1e-12) + 1e-15)


if __name__ == "__main__":
    unittest.main()
