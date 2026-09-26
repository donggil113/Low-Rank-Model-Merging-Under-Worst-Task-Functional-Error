# STATUS: P2 Low-Rank Model Merging Under Worst-Task Functional Error

최종 갱신: 2026-09-26 (UTC)

| 구분 | 상태 |
|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS**: 단위 테스트 46개 통과, CPU fixture 실행 COMPLETED |
| 과학 | **SCIENCE_NOT_EVALUATED**: 실제 adapter 실험 없음. 합성 fixture 결과는 증거가 아님 |
| 실제 adapter 파일럿 | **BLOCKED**: preflight 차단 요인 20개(승인 3, 의존성 5, 라이선스 7, revision 미고정 5) |
| 신규성 | **미확인**: 주장별 상태는 `PRIOR_ART.md` §3 참고 |

## 0. 시작 시점 저장소 상태

- 저장소는 비어 있었다(커밋 0개, 파일 없음). README, STATUS, 기존 결정, ARCHIVE_METHOD, 보류 결정, 데이터, 실행 기록이 하나도 없었다. 따라서 보존하거나 해제할 기존 결정도 없다.
- 실행 환경은 Python 3.11.15, Linux, CPU 4개다. **numpy, scipy, torch, transformers, peft, datasets, pytest가 모두 설치되어 있지 않다.** 지시에 따라 설치하지 않았고, 구현 전체를 표준 라이브러리로 작성했다.

## 1. 고정한 연구 질문

**RQ.** 같은 base 모델의 T개 LoRA adapter를 병합한다. 병합 결과는 같은 최종 rank k를 가지며, 모든 방법은 같은 calibration 입력에 접근한다. 이 조건에서 병합 delta M을 calibration 상의 최대 과제 오차 max_t e_t(M)가 최소가 되도록 고른다고 하자. e_t는 층별 상대 기능 오차다. 이 minimax 해가 다음 두 비교 대상보다 held-out test에서 **worst-task 성능**을 최소관심효과(MEI) 이상 높이는가? 이때 평균 성능은 MEI 이상 떨어지지 않아야 한다.

- 비교 대상 ①: 고정 균등 가중 WRRR. 목적함수는 같고 가중치만 고정한 형태다.
- 비교 대상 ②: 기존 기준선. TA+SVD, KnOTS, CtM, RegMean 계열을 뜻한다.

**반증 조건.** 파일럿 config `configs/pilot_4task_flan_t5_base_glue.json`에 실행 전에 고정해 두었다.

- worst-task normalized accuracy 개선의 seed 평균이 2.0pp 미만이거나, 3개 seed 중 하나라도 2.0pp 미만이면 **NOT_SUPPORTED**로 판정한다.
- mean normalized accuracy가 1.0pp를 넘게 떨어지면 **NOT_SUPPORTED**로 판정한다.
- 층별 기능 오차만 개선되는 경우에는 확장을 중단한다.
- rank, calibration 데이터, 계산을 더 써야만 이기는 경우에도 확장을 중단한다.
- 비교 대상은 **같은 calibration 접근**을 쓰고 같은 rank로 맞춘 기준선이다. 이 가운데 최선의 것을 dev 성능으로 고르며, test로 고르지 않는다.

**목적함수에 깔린 가정.** 결과를 해석할 때 모두 고려해야 한다.

1. 층마다 독립적으로 푼다. 층간 상호작용은 무시한다. RegMean++는 이 문제를 activation 출처로 다룬다.
2. 오차는 선형층 출력의 변화 ‖(M−D_t)X_t‖²이다. 비선형성 이후의 영향이나 과제 손실과 같지 않다.
3. X_t는 과제 t의 fine-tuned 모델이 과제 t의 calibration 입력에서 내는 activation이다. 이는 RegMean의 관행을 따른 것이다.
4. 상대 정규화 e_t(M)/e_t(0)를 쓴다. 분모는 calibration 분할에서 계산한다. 과제별로 스케일을 고정해 가중하는 셈이며, 이런 가중은 IterIS 같은 선행연구에 이미 있다.
5. Σ_t를 n_cal개 표본으로 추정한다. max 목적은 추정 잡음에 민감할 수 있다(§4의 fixture 관찰 참고).
6. D_t=I이면 문제는 min-max reconstruction fair PCA와 같다. 이 경우 문제는 NP-hard이고 dual gap이 0이 아닐 수 있다. 따라서 **전체 solver가 전역 최적이라고 주장하지 않는다.**

## 2. 비교군의 정보 접근 (CPU fixture 기준)

| 방법 | calibration 입력 | 과제 식별(calibration) | dev 튜닝 | factor 사용 | 최종 rank |
|---|---|---|---|---|---|
| zero_base_model | – | – | – | – | 0 |
| ta_svd | – | – | ✓ (coef) | – | k |
| knots_ta_trunc, knots_ties_trunc (비공식) | – | – | ✓ | – | k (KnOTS 자체에는 rank 예산이 없고, 절단은 우리가 추가함) |
| ctm_like_ta (비공식, 단순화) | – | – | ✓ | – | k |
| regmean_full_rank | ✓ | ✓ | ✓ (α) | – | **12 (예산 초과. 순위 비교에서 제외)** |
| regmean_euclid_trunc / regmean_whitened_trunc | ✓ | ✓ | ✓ (α) | – | k |
| wrrr_uniform_rel | ✓ | ✓ | – | – | k |
| minimax_rel | ✓ | ✓ | – | – | k |
| factor_average_diag | – | – | – | ✓ | k (gauge 비불변 진단용) |

dev 튜닝은 기준선에만 추가 정보를 준다. 따라서 이 설정은 기준선에 유리한 쪽으로 보수적이다. 튜닝한 후보 수(grid 크기)는 raw log에 기록했다. 실제 계산 비용도 기록했다. minimax는 seed당 WRRR를 약 340번 풀었고 3.4–3.7초가 걸렸다. wrrr_uniform은 약 0.011초였다. 두 방법에 **같은 비용 상한을 주더라도 실제로 쓰는 계산량은 같지 않다.**

## 3. 구현과 검증 (TECHNICAL_TEST_PASS의 근거)

실행 명령과 결과는 다음과 같다.

- `python3 -m unittest discover -s tests -v`: 46 tests, OK, 약 9.6초.
- 로그: `runs/unittest_*/unittest.log`

| 요구 | 구현 | 검증 |
|---|---|---|
| (1) factor가 아니라 effective delta W=B@A 기준 | `adapters.py`. 모든 방법이 `effective_delta()`를 입력으로 받는다 | gauge 불변성 테스트 |
| (2) 과제별 기능 오차와 minimax 목적 | `objectives.py`, `minimax.py` | Gram 형식과 원시 표본 형식이 일치함 |
| (3) 고정 가중 WRRR, 그리고 singular/ridge/PD의 구분 | `wrrr.py`의 `pd`/`pinv`/`ridge` | 아래 표 |
| (4) 독립 reference와 알려진 특수해 | `references.py`와 테스트 | 아래 표 |
| (5) (B@R, R⁻¹@A) 불변성과 최종 rank | `random_gauge`, `numerical_rank` | 아래 표 |
| (6) cal/test 분리, 과제별 결과, 파라미터 accounting | `synthetic.py`, `evaluate`, `param_account` | 분할 스트림이 독립적이고 hash가 안정적임 |

**3a. 고정 가중 WRRR.** 상대 오차 1e-9~1e-10 수준(스케일 보정 포함)에서 다음과 일치했다.

- Izenman 원시 데이터 RRR(OLS→fitted 값 SVD 경로, 무작위 6개 인스턴스, 모든 k).
- d_out=2, k=1 grid 탐색.
- 다중 시작 ALS. closed form이 ALS에 진 경우는 없었다.

알려진 특수해도 모두 재현했다.

- T=1이고 k≥rank이면 정확히 복원된다.
- T=1, Σ=I이면 Eckart–Young과 같다.
- Σ_t=I이고 가중이 균등하면 TA 뒤에 SVD 절단을 한 것과 같다.
- full rank이면 λ-가중 RegMean과 같다.
- k=0이면 해가 0이다.
- 해가 비유일한 절단은 따로 표시한다.

**3b. singular S의 세 모드는 목적함수가 서로 다르다.**

| 모드 | 동작 |
|---|---|
| `pd` | S가 양의 정부호가 아니면 SingularGramError로 **거부**한다 |
| `pinv` | **원래 목적함수**의 전역 최소해를 준다. 부분공간 reference와 일치한다. null(S) 방향에서는 0으로 작용한다. calibration 데이터로는 null(S) 성분을 식별할 수 없다(Y·P_null을 더해도 목적값이 같음) |
| `ridge` | **다른 목적함수**(+ε‖M‖²)를 푼다. 원래 목적값은 pinv보다 크거나 같다. ε→0이면 pinv 해로 수렴한다. dual 하한 계산에는 **사용하지 않는다**(`dual_ascent`가 거부함) |

C가 null(S) 방향으로 새면 목적함수가 아래로 무한히 내려간다. 이 경우를 감지해 예외를 낸다.

**3c. minimax.** 모든 경우에 하한 ≤ reference ≤ 상한이 성립했다.

| 사례 | 결과 |
|---|---|
| 공개된 non-zero gap 인스턴스(Tantipongpipat et al. 2019, Lemma 6.2, **원문 직접 확인**) | LB = 2.25 = 4−7/4 (SDP 완화값), UB = 2.470588 ≈ 4−26/17 (정확해), M ≈ [[16,4],[4,1]]/17 |
| D_t=I, T=2 (같은 논문 Thm 1.2: SDP가 exact) | gap < 1e-6 |
| 대칭 두 과제 예제(자체 유도, OPT=3/4) | λ=(½,½)에서 절단이 비유일함. local refinement로 3/4에 도달 |
| T=1 | Eckart–Young과 같음 |
| full rank(볼록 문제, Sion 정리) | gap < 1e-6 |
| grid+ellipsoid reference, d_out=2, k=1 (테스트 3개 + 실행 로그 6개) | 모든 인스턴스에서 LB ≤ REF ≤ UB. 무작위 6개 중 2개는 dual gap이 실제로 0이 아니었다(REF−LB 상대 0.65%, 2.3%). solver 상한의 초과분 UB−REF는 최대 8.2e-5(상대)였다. **solver가 항상 최적해를 찾지는 않는다.** |

**3d. gauge 불변성과 rank.** fixture 3 seed에서 조건수 100인 무작위 R을 적용했다.

- B@A 기반 방법: 병합 M의 상대 변화가 전부 ≤ 9e-15였다.
- factor 평균 진단: 상대 변화 7.47로, 불변이 아님을 확인했다.
- 조건수 1e6인 R: 반올림 수준(< 1e-6)의 변화만 있었다.
- rank 예산 k를 지정한 방법은 모두 수치 rank ≤ k였다. regmean_full_rank는 예산을 넘는 것으로 표시했다.

## 4. CPU 합성 fixture 결과 (증거 아님)

- 실행 명령: `PYTHONPATH=src python3 -m lowrank_merge.run_cpu --config configs/cpu_synthetic.json`
- 산출물: `runs/cpu_synthetic_layer_fixture_20260926T152312Z/` (`raw_log.jsonl`, `results.json`, `summary.md`, `manifest.json`)
- git 944d9fd에서 실행했다. dirty 표시는 `runs/`가 추적되지 않아서 생긴 것뿐이다.
- config sha256 eac8a3c9…, seed별 data sha256은 manifest에 있다.
- 실측 wall 시간 58.1초, peak RSS 23.2 MiB. tracemalloc은 측정하지 않았고 NOT_MEASURED로 기록했다.
- 설정: d_out=12, d_in=16, T=4, LoRA r=4, k=4, n_cal=32/과제, n_dev=64, n_test=512, seed 0,1,2. 모든 값은 실행 전에 config에 고정했다.

| 방법 | test worst_rel (평균±sd) | test mean_rel | cal worst_rel |
|---|---|---|---|
| zero_base_model | 1.000 ± 0.000 | 1.000 | 1.000 |
| ta_svd (= knots_ta_trunc) | 1.445 ± 0.472 | 0.770 | 1.400 |
| knots_ties_trunc | 1.077 ± 0.282 | 0.770 | 0.996 |
| ctm_like_ta | 1.585 ± 0.421 | 0.835 | 1.543 |
| regmean_full_rank (rank 12, 예산 초과) | 1.341 ± 0.507 | 0.701 | 0.948 |
| regmean_euclid_trunc | 1.354 ± 0.414 | 0.734 | 1.012 |
| regmean_whitened_trunc | 1.339 ± 0.401 | 0.739 | 1.012 |
| **wrrr_uniform_rel** | **0.685 ± 0.060** | **0.588** | 0.601 |
| minimax_rel | 0.858 ± 0.179 | 0.627 | **0.472** |
| factor_average_diag | 1.254 ± 0.350 | 0.748 | 1.204 |

관찰은 다음과 같다. 합성 데이터이므로 **실제 adapter에 일반화하지 않는다.**

1. **calibration에서 minimax_rel의 dual gap은 모든 seed에서 5e-10 이하였다.** 따라서 calibration 목적에 대해서는 수치 정밀도 범위에서 최적이 인증되었다. refinement는 필요하지 않아 건너뛰었다.
2. **그런데 test worst-task에서는 minimax_rel이 wrrr_uniform_rel보다 3/3 seed 모두 나빴다(평균 +0.172).** minimax는 calibration 오차를 네 과제에서 정확히 같게 맞춘다. 이때 calibration 오차가 낮게 추정된 과제 2(gain 0.6)의 가중치가 작아진다. 그 결과 test에서 과제 2의 오차가 0.713, 0.802, 1.057로 크게 나빠졌다. **max 목적이 작은 calibration 표본(n_cal=32, d_in=16)의 추정 잡음에 과적합되었다는 해석과 일치한다.** 이것은 이 fixture에서 나타난 **초기 부정 신호**다.
3. wrrr_uniform_rel과 regmean_whitened_trunc는 같은 WRRR이다. 차이는 과제별 상대 정규화 여부와 α뿐이다. 그런데 test worst는 0.685 대 1.339로 큰 차이가 났다. 이 fixture에서는 **상대 정규화(이미 알려진 과제별 가중)가 효과의 대부분을 설명한다.**
4. data-free 기준선은 과제별 입력 공분산을 쓰지 않는다. 이 fixture는 과제마다 공분산을 크게 다르게 만들었기 때문에 구성상 data-aware 방법에 유리하다. 따라서 기준선 대비 격차를 방법의 우위로 해석하지 않는다.

**파일럿 전에 필요한 조치.** minimax를 정규화하는 방안이 있다. 예를 들어 λ를 균등 분포 쪽으로 수축하거나 KL 반경을 제한하는 방식이다. 이런 변형을 도입하려면 먼저 사전 등록하고 dev에서 골라야 한다. 이번 단계에서는 **구현하지 않았다.** test를 본 뒤에 방법을 바꾸는 것이기 때문이다.

## 5. 실행 기록 (실패와 중단 포함)

| 실행 | 결과 |
|---|---|
| 첫 timing smoke 실행(scratchpad, tracemalloc 켬, 1 seed) | 약 4.5 CPU-분이 지나도 seed 0이 끝나지 않아 **직접 중단(kill)** 했다. 결과로 쓰지 않는다. 원인은 dual gap이 이미 닫혔는데도 block refinement를 돌린 것과 tracemalloc 오버헤드였다. 조치로 gap이 닫혔으면 refinement를 생략하고, tracemalloc은 opt-in으로 바꿨다. |
| profiling 실행(seed 0, 결과 파일 없음) | recorded run 전에 seed 0의 test 지표를 미리 보았다. 그 뒤로 config와 방법은 **바꾸지 않았다.** 이후 변경은 테스트 버그 수정과 summary 표 형식뿐이다. |
| recorded CPU run | COMPLETED, 58.1초 |
| pilot preflight `runs/pilot_preflight_20260926T152312Z/` | BLOCKED, exit 2 |
| 알려진 결함 | recorded run의 `summary.md`에서 "final rank" 열이 dict로 출력되었다. 코드는 나중에 고쳤다. 기준 데이터는 `results.json`이며, run은 다시 돌리지 않았다. |

## 6. 파일럿 (실행하지 않음)

- config: `configs/pilot_4task_flan_t5_base_glue.json`
- 실행 명령: `PYTHONPATH=src python3 -m lowrank_merge.pilot --config configs/pilot_4task_flan_t5_base_glue.json`
- 후보 base: google/flan-t5-base. 라이선스는 Apache-2.0로 보고되었으나 미검증이다.
- 후보 adapter: tanganke/flan-t5-base_glue-{cola,mrpc,rte,sst2}_lora-16 (r=16, α=32, q/v). **라이선스 태그가 없다.** declared base는 cola와 rte만 서브에이전트가 확인했다.
- 분할: calibration은 GLUE train 128개/과제이며 label을 쓰지 않는다. dev와 test는 official validation을 seed 0으로 반씩 나눈다.
- task head는 없다(공유 LM head). 추론에서는 과제별 prompt template을 쓰므로 **task identity를 사용한다.** 병합에서도 과제별 Gram을 만들기 위해 task identity를 사용한다.
- primary metric은 test worst-task normalized accuracy(각 adapter 단독 대비)이다. MEI는 +2.0pp이고, mean 하락 허용치는 1.0pp다. seed는 calibration 표본 {0,1,2}다. 자원 상한은 CPU 6시간, GPU 0, RAM 12GB, 디스크 3GB다.
- 두 번째 후보는 KnOTS ViT-B/32 LoRA(CLIP)다. 과제별 zero-shot head가 있어 추론에 task identity가 필요하고, adapter_config의 base가 null이며, 라이선스도 미검증이다.

**Blocker (정확한 목록)**

1. 승인: 가중치 다운로드, 데이터셋 다운로드, 의존성 설치가 모두 미승인이다.
2. 의존성: numpy, torch, transformers, peft, datasets가 설치되어 있지 않다. 순수 Python 구현은 768×768 층을 처리할 수 없으므로 numpy/torch 포팅이 필요하다. 포팅한 코드는 작은 행렬에서 순수 Python reference와 교차검증해야 한다.
3. 라이선스: base, adapter 4개, 데이터셋, 평가 코드의 라이선스가 모두 UNVERIFIED다. 특히 adapter에는 라이선스 표기가 없다.
4. 고정: base와 adapter 4개의 revision과 hash가 고정되어 있지 않다.
5. 오염 위험: adapter를 학습할 때 GLUE validation을 checkpoint 선택에 썼는지 확인되지 않았다. 썼다면 dev/test가 오염된다.

## 7. 미검증 주장

- 층별 worst-task 오차를 줄이면 실제 과제의 worst-task 성능이 좋아진다는 주장. 평가하지 않았다.
- A3(joint WRRR 절단)과 A6(target이 있는 minimax RRR과 dual 하한)이 선행연구에 없다는 주장. 확인한 범위에서 찾지 못했을 뿐이며 **신규성 미확인**이다.
- KnOTS-TIES와 CtM-like 구현이 공식 방법과 같다는 주장. 비공식 구현이며 공식 코드와 대조하지 않았다.
- 대부분의 논문 세부 내용은 서브에이전트가 읽은 것이다. 메인 에이전트가 원문을 직접 확인한 것은 Tantipongpipat et al. 2019의 Thm 1.2/1.3, §2 가정, Lemma 6.2뿐이다. 3편(CtM, Budget-Aware, Cao et al.)은 초록만 직접 확인했다.

## 8. 다음 단계 (각 단계에 별도 승인 필요, 이번 단계에서는 진행하지 않음)

1. 라이선스와 revision을 사람이 확인하고 값을 고정한다.
2. 의존성 설치를 승인받은 뒤 numpy/torch backend를 작성한다. 작성한 backend는 순수 Python reference와 교차검증한다.
3. 기준선(zero, TA-SVD, WRRR-uniform)의 작은 실제 실행으로 파이프라인을 확인한다.
4. minimax 과적합 대책(λ 수축 등)을 dev 기준으로 사전 등록한 뒤 파일럿을 실행한다.
