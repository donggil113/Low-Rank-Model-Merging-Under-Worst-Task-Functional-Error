# P2 stage 3: 결정론적 bound 수치 점검과 원고 v0

- 판정: 소프트웨어 **TECHNICAL_TEST_PASS**(테스트 56개), 과학 **SCIENCE_NOT_EVALUATED**, 원인 **CAUSE_UNDETERMINED**, 실제 adapter **BLOCKED**.
- 원고: `paper/main.pdf`. 공식 ICLR 2027 style로 빌드했다. 전체 16쪽이고 본문은 9쪽이다. 내부 working draft이며 제출하지 않았다.

## 비교 대상 확정 (원자료 field 기준)

- **0.713/0.802/1.057**: minimax 해의 worst-task 오차를 test sample에서 잰 값이다(이미 본 것).
  - 출처: stage-1 `summary.test.worst_rel` = stage-2 `max(test_rel_per_task_PREVIOUSLY_SEEN)`, arm `emp_moment__emp_norm`, draw `orig_seen`.
- **0.726/0.768/1.081**: **같은 minimax 해**를 population 목적 `F_pop`으로 잰 값이다.
- 따라서 두 triple은 sample 평가와 population 평가의 비교이며, 두 방법의 비교가 아니다.
- stage 1과 stage 2의 병합 행렬은 전체 SHA-256이 6/6 일치한다(bit-identical).

## Bound 점검 (`runs/p2_stage3_bound_check_20260926T225217Z/`)

- 실행 명령: `PYTHONPATH=src python3 -m lowrank_merge.run_bound_check --config configs/p2_stage3_bound_check.json`
- config는 실행 전 커밋 `8428292`에서 고정했다.
- 방식: stage 2가 행렬을 hash로만 저장했으므로 같은 코드 경로로 27개를 재생성했다.
  - SHA-256이 27/27 일치했고, 재생성한 F_pop과 기록값의 차이는 0.0이다.
  - 새 실험이 아니라 기록된 인스턴스를 재분석한 것이다.
- 결과: 모든 부등식이 성립했다. excess bound 24/24, pointwise Lemma 1/3 점검 192/192.
- **그러나 이 bound들은 정보가 없다(vacuous).**
  - 상대 moment 오차 max_k‖S−Ŝ‖/‖S‖는 0.64–1.04다.
  - 가장 작은 bound가 14.0(상대오차 단위)으로, 실제 excess의 61배 이상이다.
  - pointwise 부등식도 우변의 최대 27%만 사용한다.
  - 즉 관측과 모순은 없지만, 관측된 차이의 크기나 방향을 설명하지 못한다.
- 자원: CPU 216.1초(RLIMIT 1,100초, 과제 상한 1,200초), wall 217.6초, peak RSS 23.4 MiB.

## 원고와 빌드

- 도구: tectonic을 받을 수 없어(GitHub release 403) Ubuntu 서명 저장소의 최소 TeX Live를 설치했다.
  - 1차 시도는 실패했고 로그를 보존했다. 2차 시도에서 성공했다.
  - 설치 기록은 `paper/BUILD.md`에 있다.
- 빌드 결과: 경고 0개(underfull 8개 제외), 빌드와 렌더링 CPU 합계 14.9초 / 600초.
- 정정 두 건을 반영했다(`CORRECTIONS.md`).
  - 0.858은 이중 반올림한 값이고 정확히는 0.857이다.
  - 최대 |z| 3.03은 로그 끝줄만 읽은 값이고 실제로는 3.38이다.
