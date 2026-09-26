# 정정 이력 (Correction history)

원래 문서는 수정하거나 삭제하지 않고 그대로 보존한다. 정정은 이 파일과 원고의 Appendix
"Correction history"에만 기록한다. 판단 기준은 raw result 파일이다.

| # | 원래 기재 (위치) | 원자료 값 (파일 · field) | 원인 | 발견 |
|---|---|---|---|---|
| 1 | "minimax_rel 0.858 ± 0.179"<br>(`STATUS.md` §4 표, `reports/p2_stage2_diagnostic.md` "기존 주장") | 0.8574992618…<br>(`runs/cpu_synthetic_layer_fixture_20260926T152312Z/results.json` · `per_method.minimax_rel.test_worst_rel.mean`)<br>소수 셋째 자리로는 **0.857** | `summary.md`에 표시된 "0.8575"를 다시 반올림함 (이중 반올림) | 2026-09-26, 원고 exporter |
| 2 | "최대 \|z\|는 3.03"<br>(`reports/p2_stage2_diagnostic.md` 설계 항목) | 3.383, teacher 0 · task 2<br>(`runs/p2_stage2_moment_normalizer_diag_20260926T155649Z/results.json` · `max(mc_moment_check[].worst_abs_z)`) | 로그의 마지막 줄만 읽음 | 2026-09-26, 원고 exporter |
| 3 | "0.713/0.802/1.057 대 0.726/0.768/1.081"<br>(stage-2 요약의 나란한 표기) | 두 triple 모두 **같은 minimax 해**다. 앞은 test sample(이미 본 것) 평가, 뒤는 population F 평가다. 두 방법의 비교가 아니다.<br>stage-1 `summary.test.worst_rel`과 stage-2 `max(test_rel_per_task_PREVIOUSLY_SEEN)`이 같은 값이고, 병합 행렬 SHA-256도 일치한다. | 요약에서 비교 대상을 명시하지 않음 | 2026-09-26, 원자료 field 대조 |
| 4 | "calibration 오차가 낮게 추정된 과제 2의 가중치가 작아진다"는 기제 설명<br>(`STATUS.md` §4) | oracle Pop/Pop 해에서도 과제 2의 λ가 작다: 0.165 (teacher 0), 0.188 (teacher 2)<br>(stage-2 `results.json` · `rows[arm=pop_moment__pop_norm].audit.lambda_star[2]`) | 해석이 과했음 | stage 2 (STATUS §4에 이미 정정 주석 있음) |
