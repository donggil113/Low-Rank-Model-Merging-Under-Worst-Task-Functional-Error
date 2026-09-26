# P2 stage 2: empirical minimax 일반화 실패의 원인 분리 (개발 진단)

- 판정: 소프트웨어 **TECHNICAL_TEST_PASS**. 과학 **SCIENCE_NOT_EVALUATED**. 실제 adapter **BLOCKED**.
- 성격: 합성 fixture에서 수행한 **개발 진단**이다. 확증 실험이 아니다.
  - 원래 calibration draw의 test 결과는 이미 본 상태다.
  - population arm은 oracle 진단이며, 실제로 배포할 수 있는 방법이 아니다.
- config: `configs/p2_stage2_moment_normalizer_diag.json`. 실행 전인 커밋 `06947af`에서 고정했다.
- 실행 명령: `PYTHONPATH=src python3 -m lowrank_merge.run_diag --config configs/p2_stage2_moment_normalizer_diag.json`
- 원자료: `runs/p2_stage2_moment_normalizer_diag_20260926T155649Z/` (`raw_log.jsonl`, `results.json`, `manifest.json`)
- 파생 표: `python3 scripts/analyze_p2_stage2.py <run_dir>`로 만들며, 출력은 `reports/p2_stage2_analysis_output.txt`에 있다. 이 파일은 재생성한 파생 로그다.
- 실행 규모: 78개 cell 모두 OK, NOT_RUN과 FAILED는 0개.
  - 구성: teacher 3개 × (draw 4개 × arm 3개 + pop/pop 1개) × 방법 2개.
- 자원: CPU 285.8초(상한 1,800초), wall 288.5초, peak RSS 46.9 MiB(상한 3 GiB), worker 1개, thread 1개.
  - 설치와 다운로드는 하지 않았다.
  - 상한은 RLIMIT_CPU와 RLIMIT_AS를 프로세스 안에서 걸어 강제했다.

## 기존 주장

- 이전 stage의 합성 fixture 결과: test worst_rel이 minimax_rel 0.858, uniform WRRR 0.685였다. 3개 seed 모두에서 minimax가 더 나빴다.
- 이전 STATUS의 해석: "calibration 표본 잡음에 과적합했다는 해석과 일치한다." 당시에는 원인을 확정하지 않았다.

## 이번 관찰

**설계**

- teacher: 기록된 fixture의 문제 seed 0, 1, 2를 그대로 썼다. 데이터 hash가 일치하는지 확인했다.
  - teacher 3개는 서로 다른 문제다. 같은 문제의 독립 seed로 세지 않는다.
- calibration: 새 재추출 seed 101, 102, 103을 썼고, 과제당 n_cal=32다. 원래 draw는 `orig_seen`으로 따로 보고한다.
- 평가: 모든 결과를 같은 population worst-task 상대오차 F_pop으로 평가했다.
  - population 2차 모멘트는 L_tL_tᵀ다. 생성기의 평균이 정확히 0이기 때문이다.
  - 이 닫힌 식은 Monte Carlo 20,000 표본으로 확인했다. 12개 행렬 모두 통과했고, 최대 |z|는 3.03이다.

**결과.** Δ = F_pop(minimax) − F_pop(uniform)이다. 음수이면 minimax가 낫다. "새 draw"는 teacher 3개 × 재추출 3개, 9개 cell의 평균이다.

| arm (moment / normalizer) | 새 draw 평균 Δ (sd) | teacher별 평균 Δ | 사전 규칙 판정 | orig_seen Δ (teacher 0/1/2) |
|---|---|---|---|---|
| empirical / empirical | **+0.0065** (0.078) | −0.042 / −0.014 / +0.075 | AMBIGUOUS | +0.069 / +0.113 / +0.301 |
| empirical / population | **+0.039** (0.082) | +0.034 / −0.002 / +0.086 | PROBLEM_PRESENT | −0.053 / +0.177 / +0.173 |
| population / empirical | **−0.025** (0.043) | −0.041 / −0.025 / −0.010 | PROBLEM_ABSENT | −0.009 / −0.050 / +0.117 |
| population / population (oracle, draw와 무관) | **−0.093** | −0.066 / −0.138 / −0.076 | PROBLEM_ABSENT | (같은 값) |

- **원래 draw에서의 실패는 평가 방식의 산물이 아니다.** population 평가에서도 그대로 재현되었다.
  - 이전에 본 test 표본 worst: minimax 0.713 / 0.802 / 1.057, uniform 0.642 / 0.660 / 0.754.
  - population 평가: minimax 0.726 / 0.768 / 1.081, uniform 0.657 / 0.655 / 0.780.
- **새 draw에서는 실패의 크기가 재현되지 않았다.** empirical/empirical 평균 Δ는 +0.0065이고 cell 간 sd는 0.078이다. 원래 draw의 큰 격차는 draw에 따른 변동이 크게 작용한 결과로 보인다.
- **calibration 낙관 편향은 minimax 쪽이 훨씬 크다.** 새 draw, empirical/empirical 기준으로 fit 목적의 max가 F_pop과 얼마나 차이 나는지 비교했다.

  | 방법 | fit max | F_pop | 차이 |
  |---|---|---|---|
  | minimax | 0.497 | 0.672 | +0.175 |
  | uniform | 0.606 | 0.665 | +0.059 |

- **정규화 분모가 크게 흔들린다.** 과제별 empirical/population 분모 비율은 0.72에서 1.73 사이다.
- λ*를 보면, oracle(pop/pop)도 과제 2에 작은 가중치를 준다(0.165, 0.188). 따라서 "과제 2의 가중치가 작다"는 사실 자체는 잡음 때문이 아니다.
  - 이전 STATUS의 기제 설명 가운데 이 부분은 고쳐야 한다. 잡음 때문에 생긴 것은 λ와 M이 oracle에서 벗어난 정도다.

## 반박된 설명

이 인스턴스들에 한정한 수치적 근거다. 일반적인 증명이 아니다.

**1. solver 실패: 반박됨.** minimax 적합 39개 전부에서 다음이 성립했다.

- 계산한 bound의 상대 gap이 2.0e-9 이하였다.
  - LB는 g(λ)를 λ*에서 계산한 값이다. inner 문제는 pd 모드의 WRRR이다.
  - S의 null 공간은 없었고, 조건수는 19.3 이하였다.
  - λ*에서 해는 유일했다(σ_k > σ_{k+1}).
- 독립 기준인 원시 데이터 Izenman RRR로 계산한 g와의 상대 차이가 2.5e-15 이하였다.
- UB를 원시 표본으로 다시 계산했을 때의 상대 차이가 2.8e-15 이하였다.
- 해 M의 rank 조건을 만족했다(σ_{k+1}/σ_1 ≤ 2e-16).
- λ는 simplex 내부에 있었다(최솟값 0.13).
- 이 값들은 float 계산 결과이므로 엄밀한 인증이 아니다.
- 코드 감사에서 확인한 점도 있다.
  - LB는 inner WRRR에 넘긴 것과 **같은 λ**로 계산한다.
  - ridge 모드는 bound 계산에서 거부된다.
  - LB는 quadratic form에 상쇄가 있으므로 부동소수 오차 범위 안에서만 유효하다. 그래서 적합마다 원시 데이터 reference와 대조했다.

**2. evaluator 또는 목적함수 구현의 불일치: 반박됨.**

- population/population arm에서는 fit 목적과 평가 목적이 1e-15 수준까지 같다. 이 arm에서 minimax는 teacher 3개 모두에서 uniform보다 나았다(Δ −0.066 / −0.138 / −0.076).
- 이전에 본 test 표본 평가도 population 평가와 방향과 크기가 일치했다.

**3. 정규화 분모의 추정 잡음만으로 설명하는 것: 뒷받침되지 않음.**

- 분모를 population 값으로 바꾸고 moment는 empirical로 둔 arm에서 문제가 오히려 PROBLEM_PRESENT(+0.039)였다.

## 남은 설명 (미확정)

- **empirical moment 추정 잡음(과제당 n_cal=32로 16×16 Gram을 추정)이 가장 유력한 후보다.**
  - population moment를 넣은 두 arm에서만 평균 불이익이 사라졌다. 분모가 empirical이든 population이든 마찬가지였다.
  - 다만 사전 규칙상 원인 귀속은 **성립하지 않는다.** 규칙은 "empirical moment arm들에서 PROBLEM_PRESENT"를 요구하는데, empirical/empirical이 AMBIGUOUS였다. 따라서 공식 판정은 **CAUSE_UNDETERMINED**다.
- 분모 잡음도 기여할 가능성이 있다.
  - population/empirical arm에서 minimax의 이득은 −0.093에서 −0.025로 줄었다.
  - teacher 2의 원래 draw에서는 분모 잡음만으로 +0.117이 나왔다.
- empirical/population arm이 empirical/empirical보다 나빴다. 같은 draw에서 나온 moment 오차와 분모 오차가 비율 안에서 일부 상쇄된다는 가설과 맞는다. 이 가설은 **검증하지 않았다.**
- 실용적인 결론은 다음과 같다.
  - calibration n=32에서 empirical minimax는 draw에 따른 분산이 크다.
  - 평균적으로도 uniform WRRR보다 낫지 않았다(+0.0065).
  - 따라서 **uniform WRRR를 주 기준선으로 유지한다.**
  - 이는 합성 문제에서의 관찰이다. 실제 병합 정확도에 대해서는 아무것도 말해 주지 않는다.
- 이번 단계에서 하지 않은 것: shrinkage, n_cal/rank/task 수 sweep, 절대오차 목적과의 비교(상대 정규화 자체의 효과). 따라서 "상대오차 정규화라는 설계 선택 자체"의 영향은 **분리하지 못했다.**

## 실제 자산 blocker

읽기와 metadata 확인만 했다. 근거 파일은 `runs/p2_stage2_adapter_admission_20260926T155226Z/admission.json`과 `raw/`다.

- **확인한 사항**
  - adapter 4개의 설정은 모두 같다: base `google/flan-t5-base`, r=16, α=32, dropout 0.1, target q/v, SEQ_2_SEQ_LM.
  - repo revision: cola `0e13ad9`, mrpc `920cbe7`, rte `20a3b5a`, sst2 `84371dd`.
  - base는 apache-2.0 tag이며 현재 sha는 `7bcac57`이다.
- **사용권**: adapter repo에 license tag가 없다. 모델 카드는 자동 템플릿이고 License 항목이 "[More Information Needed]"다. 따라서 **사용권이 불명**하다.
- **base revision**: adapter_config의 `revision: null`이어서 학습에 쓴 base revision이 기록되어 있지 않다.
- **학습 split과 checkpoint 선택**: 모델 카드에 정보가 없다. FusionBench HEAD `54c9e8c`에서도 이 adapter들의 학습 recipe를 찾지 못했다. 따라서 **dev/test가 학습과 독립인지 확인할 수 없다.**
  - FusionBench는 GLUE **validation**으로 평가한다. 이는 우리가 dev/test로 계획한 바로 그 split이다.
- **metric**: FusionBench는 4개 과제 모두 생성된 label 단어의 **exact-match accuracy**로 평가한다. CoLA의 Matthews 상관이나 MRPC의 F1이 아니다.
  - 게시자가 보고한 LoRA-CoLA 성능은 CoLA 69.1로, 사전학습 base와 같다. 그래서 CoLA에 대해서는 "adapter 대비 정규화 정확도"가 정보를 주지 못한다.
  - 69.1이 다수 클래스 비율과 비슷하다는 점은 **UNVERIFIED**다.
- **조치**: 이 자산들은 이번 실험 입력에서 제외했고 **BLOCKED**를 유지한다. 다른 adapter를 탐색하거나 직접 학습하지 않았다.
- **파일럿 config와의 관계**: 기존 파일럿 config(`configs/pilot_4task_flan_t5_base_glue.json`, 보존함)는 "adapter 단독 대비 normalized accuracy"를 primary metric으로 둔다. 위 사실 때문에 CoLA 항목은 이 metric으로 해석할 수 없다. config를 고쳐야 하지만 이번 단계에서는 **수정하지 않았다.**
