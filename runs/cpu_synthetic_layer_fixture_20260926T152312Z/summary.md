# CPU synthetic fixture: cpu_synthetic_layer_fixture

Implementation check on synthetic data; NOT evidence about real adapters.

k = 4, seeds = [0, 1, 2], primary metric = test worst-task relative functional error (max_t e_t(M)/e_t(0) on test inputs)

| method | final rank | test worst_rel (mean±sd) | test mean_rel | cal worst_rel | access (cal / dev-tuning / factors) |
|---|---|---|---|---|---|
| zero_base_model | {'mean': 0.0, 'sd': 0.0, 'values': [0, 0, 0]} | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | False / False / False |
| ta_svd | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.4445 ± 0.4722 | 0.7695 | 1.3997 | False / True / False |
| knots_ta_trunc | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.4445 ± 0.4722 | 0.7695 | 1.3997 | False / True / False |
| knots_ties_trunc | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.0771 ± 0.2816 | 0.7699 | 0.9959 | False / True / False |
| ctm_like_ta | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.5853 ± 0.4213 | 0.8348 | 1.5427 | False / True / False |
| regmean_full_rank | {'mean': 12.0, 'sd': 0.0, 'values': [12, 12, 12]} | 1.3412 ± 0.5065 | 0.7009 | 0.9479 | True / True / False |
| regmean_euclid_trunc | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.3535 ± 0.4144 | 0.7342 | 1.0115 | True / True / False |
| regmean_whitened_trunc | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.3386 ± 0.4009 | 0.7391 | 1.0121 | True / True / False |
| wrrr_uniform_rel | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 0.6854 ± 0.0597 | 0.5879 | 0.6008 | True / False / False |
| minimax_rel | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 0.8575 ± 0.1785 | 0.6274 | 0.4717 | True / False / False |
| factor_average_diag | {'mean': 4.0, 'sd': 0.0, 'values': [4, 4, 4]} | 1.2535 ± 0.3496 | 0.7475 | 1.2036 | False / False / True |

## Paired test worst_rel difference (minimax_rel minus method; negative = minimax better)

- zero_base_model: mean -0.1425, minimax better in 2/3 seeds
- ta_svd: mean -0.5870, minimax better in 3/3 seeds
- knots_ta_trunc: mean -0.5870, minimax better in 3/3 seeds
- knots_ties_trunc: mean -0.2196, minimax better in 3/3 seeds
- ctm_like_ta: mean -0.7278, minimax better in 3/3 seeds
- regmean_full_rank: mean -0.4837, minimax better in 3/3 seeds
- regmean_euclid_trunc: mean -0.4960, minimax better in 3/3 seeds
- regmean_whitened_trunc: mean -0.4811, minimax better in 3/3 seeds
- wrrr_uniform_rel: mean +0.1721, minimax better in 0/3 seeds
- factor_average_diag: mean -0.3960, minimax better in 3/3 seeds

## Gauge (B R, R^-1 A) invariance: max relative change of merged M over seeds

- zero_base_model: 0.00e+00
- ta_svd: 3.33e-15
- knots_ta_trunc: 4.06e-15
- knots_ties_trunc: 8.80e-15
- ctm_like_ta: 5.83e-15
- regmean_full_rank: 3.32e-15
- regmean_euclid_trunc: 4.28e-15
- regmean_whitened_trunc: 3.91e-15
- wrrr_uniform_rel: 5.97e-15
- minimax_rel: 5.75e-15
- factor_average_diag: 7.47e+00

## Singular combined Gram (inputs in a subspace)

- mode=pd ridge=0.0: REFUSED_AS_DESIGNED
- mode=pinv ridge=0.0: OK, cal objective 0.459288, exact_for_original=True
- mode=ridge ridge=0.1: OK, cal objective 0.667859, exact_for_original=False
- mode=ridge ridge=0.001: OK, cal objective 0.459786, exact_for_original=False
- mode=ridge ridge=1e-05: OK, cal objective 0.459288, exact_for_original=False

## Reference checks (d_out=2, k=1)

- instance=tantipongpipat2019_lemma6.2, lower_bound=2.25, expected_relaxation=2.25, upper_bound=2.47059, expected_exact=2.47059
- instance=random_0, d_out=2, d_in=3, n_tasks=2, lower_bound=0.356253, reference_grid_ellipsoid=0.356253, upper_bound=0.356253, lb_le_ref=True, ref_le_ub=True, ub_minus_ref_rel=4.67459e-16, ref_minus_lb_rel=-3.11639e-16, solver_s=0.0449856, reference_s=2.33445
- instance=random_1, d_out=2, d_in=3, n_tasks=2, lower_bound=0.514689, reference_grid_ellipsoid=0.514689, upper_bound=0.514689, lb_le_ref=True, ref_le_ub=True, ub_minus_ref_rel=6.39249e-12, ref_minus_lb_rel=-1.07854e-15, solver_s=0.0446262, reference_s=2.34035
- instance=random_2, d_out=2, d_in=2, n_tasks=3, lower_bound=0.792088, reference_grid_ellipsoid=0.797247, upper_bound=0.797247, lb_le_ref=True, ref_le_ub=True, ub_minus_ref_rel=1.56703e-07, ref_minus_lb_rel=0.00647108, solver_s=0.464104, reference_s=1.75233
- instance=random_3, d_out=2, d_in=3, n_tasks=2, lower_bound=0.505346, reference_grid_ellipsoid=0.505346, upper_bound=0.505346, lb_le_ref=True, ref_le_ub=True, ub_minus_ref_rel=4.39392e-16, ref_minus_lb_rel=-8.78783e-16, solver_s=0.0451835, reference_s=2.39822
- instance=random_4, d_out=2, d_in=3, n_tasks=4, lower_bound=0.807206, reference_grid_ellipsoid=0.807206, upper_bound=0.807206, lb_le_ref=True, ref_le_ub=True, ub_minus_ref_rel=2.36868e-08, ref_minus_lb_rel=2.33816e-15, solver_s=0.632337, reference_s=3.50956
- instance=random_5, d_out=2, d_in=3, n_tasks=4, lower_bound=0.729276, reference_grid_ellipsoid=0.74642, upper_bound=0.746482, lb_le_ref=True, ref_le_ub=True, ub_minus_ref_rel=8.22632e-05, ref_minus_lb_rel=0.0229683, solver_s=0.782491, reference_s=3.37092

Wall time 58.1s, peak RSS 23.2 MiB, git 944d9fda8a3301501c673e4f576db83706bcb1e8 (dirty=True)
