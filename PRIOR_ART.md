# Prior-art 대조표 (P2: Low-Rank Model Merging Under Worst-Task Functional Error)

작성일: 2026-09-26. 검색 범위는 웹 검색과 arXiv/OpenReview/학회 페이지이며, 2026년 arXiv 논문은 빠진 것이 있을 가능성이 높다.
"찾지 못함"은 **확인한 출처 안에서 찾지 못했다는 뜻이다**. 신규성을 인증하는 말이 아니다.

## 검증 상태 표기

| 표기 | 의미 |
|---|---|
| `FULL_TEXT_READ_SELF` | 이 세션의 메인 에이전트가 해당 절의 원문을 직접 읽음 |
| `FULL_TEXT_READ_SUBAGENT` | 웹을 읽는 서브에이전트가 method/theory 절 원문(arXiv HTML 또는 PDF)을 읽었다고 보고함. 인용은 메인 에이전트가 모두 재확인하지 않음 |
| `PARTIAL_FETCH_SUMMARY` | 서브에이전트가 요약 도구를 거쳐서만 읽음. 원문 확인이 약함 |
| `ABSTRACT_SELF` | 메인 에이전트가 초록과 메타데이터만 직접 확인함 |
| `FULL_TEXT_UNVERIFIED` | 초록, 메타데이터, 2차 출처만 확인함 |

## 1. 가장 가까운 원논문과 후속 연구

| # | 논문 (저자, 연도, venue, 링크) | 상태 | 이미 알려진 내용 | 우리 구성요소와의 관계 |
|---|---|---|---|---|
| 1 | **KnOTS**: Model merging with SVD to tie the KnOTS. Stoica, Ramesh, Ecsedi, Choshen, Hoffman. ICLR 2025. [2410.19735](https://arxiv.org/abs/2410.19735) | FULL_TEXT_READ_SUBAGENT | 과제 update를 이어붙인 행렬의 SVD로 U를 공유하고, 과제별 V_t를 TIES/DARE로 병합한다. ΔW=B@A를 기준으로 한다(App. A). 데이터를 쓰지 않는다. 최종 rank 예산은 없다. §5.1에 따르면 선형 평균일 때 KnOTS+TA는 TA와 같다. | B@A 기준 병합은 이미 있다. 우리는 KnOTS-TA=TA를 수치로 확인했다(`test_knots_ta_equals_task_arithmetic`). rank-k 절단은 KnOTS의 일부가 아니라 우리가 공정 비교를 위해 붙인 것이다. |
| 2 | **RegMean**: Dataless Knowledge Fusion by Merging Weights of Language Models. Jin, Ren, Preotiuc-Pietro, Cheng. ICLR 2023. [2212.09849](https://arxiv.org/abs/2212.09849) | FULL_TEXT_READ_SUBAGENT (다른 서브에이전트는 FTU로 표기) | 층별로 Σ_i‖W^T X_i − W_i^T X_i‖²를 최소화하고 닫힌 해는 (ΣG_i)^{-1}ΣG_iW_i이다. α trick(off-diagonal 축소, 기본값 0.9, T5는 0.1)은 정규화를 추가하는 것과 같다(App. A). rank 제약은 없다. | 가중치 λ가 균등이고 full rank일 때 우리 WRRR 해는 λ-가중 RegMean과 같다(`test_full_rank_is_weighted_regmean`). α는 목적함수를 바꾸므로 기준선으로만 쓴다. |
| 3 | **RegMean++**. The-Hai Nguyen, Dang Huu-Tien, Takeshi Suzuki, Le-Minh Nguyen. 2025 (TMLR accepted로 보고됨). [2508.03121](https://arxiv.org/abs/2508.03121) | FULL_TEXT_READ_SUBAGENT | RegMean과 닫힌 해가 같다. 입력 activation을 병합된 모델의 이전 층에서 가져온다는 점만 다르다. | 파일럿 config에는 activation 출처 ablation으로만 넣었다. |
| 4 | **Compress then Merge (CtM)**: From Multiple LoRAs into One Low-Rank Adapter. He, Ding, Huang, Yang, Li, Huang. ICML 2026. [2606.03723](https://arxiv.org/abs/2606.03723) | ABSTRACT_SELF + FULL_TEXT_READ_SUBAGENT | T개 LoRA를 rank-r LoRA 하나로 합친다. 공유 부분공간을 Tucker-2(HOSVD+HOOI)로 구하고 r×r core에서 TIES 등으로 병합하므로 rank ≤ r가 구성상 보장된다. 가중치 공간의 Frobenius 목적만 쓰며 activation은 쓰지 않는다(서브에이전트 보고). worst-task retention은 평가 지표로만 보고한다. | 동일 최종 rank에서 가장 직접적인 data-free 경쟁자다. 여기 구현한 `ctm_like_ta`는 비공식 단순화판이다(β 정규화와 TIES core 병합을 뺐다). 보고용 비교에는 공식 코드가 필요하다. |
| 5 | **Compress then Serve**. Brüel-Gabrielsson et al. ICML 2025. [2407.00066](https://arxiv.org/abs/2407.00066) | PARTIAL_FETCH_SUMMARY | min Σ_i‖B_iA_i − UΣ_iV^T‖², LoRA마다 별도 Σ_i를 둔다(serving용). | 병합 결과가 하나가 아니므로 직접 비교 대상은 아니다. |
| 6 | **Core Space merging**. Panariello et al. NeurIPS 2025. [2509.17786](https://arxiv.org/abs/2509.17786) | FULL_TEXT_READ_SUBAGENT | 쌓은 B, A의 기저 위에 Tr×Tr core를 만든다. 무손실이며, 선형 병합이면 full-space 병합과 같다. | B@A와 동등한 표현이다. 비교 파일럿의 코드 후보(Apache-2.0로 보고됨)다. |
| 7 | **TSV-Merge** (Gargiulo et al., CVPR 2025, [2412.00081](https://arxiv.org/abs/2412.00081)), **Iso-C/Iso-CTS** (Marczak et al., ICML 2025, [2502.04959](https://arxiv.org/abs/2502.04959)) | PARTIAL_FETCH_SUMMARY / FULL_TEXT_READ_SUBAGENT | SVD 기반 과제 간섭 감소다. 여기서 "whitening"은 특이벡터 직교화이지 activation S^{1/2} 백색화가 아니다. | SVD 기반 병합은 이미 있다. |
| 8 | **LoRM** (Salami et al., ICLR 2025, [2410.17961](https://arxiv.org/abs/2410.17961)), **ESM** (Li et al., CVPR 2026, [2602.20208](https://arxiv.org/abs/2602.20208)), **IterIS** (Chen et al., CVPR 2025, [2411.15231](https://arxiv.org/abs/2411.15231)), **LOT Merging** ([2505.23859](https://arxiv.org/abs/2505.23859)) | FULL_TEXT_READ_SUBAGENT (LOT는 PARTIAL) | **LoRM**: RegMean 목적함수를 LoRA factor 하나를 고정한 채 교대로 푼다. **ESM**: 과제별로 activation-aware 절단(과제 하나에 대한 RRR 해)을 한다. **IterIS**: 과제별 가중치 λ_i=‖W_i‖²/‖W_i^TX_i‖²를 쓴다. **LOT**: pseudo-inverse RegMean이다. | calibration 기반 기능 오차와 rank의 **부분적** 결합, 그리고 **과제별 고정 가중치**는 이미 있다. |
| 9 | **Budget-Aware LoRA Merging** (Not All Ranks Are Equal). Amballa, Saidutta, Li, Valkov, Chappidi. arXiv 2026-09. [2609.22237](https://arxiv.org/abs/2609.22237) | ABSTRACT_SELF + FULL_TEXT_READ_SUBAGENT | 서브에이전트 보고에 따르면 Σ_i‖ΔW_i−ΔW_m‖²_{G_i}(Eq. 2)와 과제별 상대 정규화에서 출발한다. 이후 G_i를 data-free proxy로 바꾸고, 전역 rank 예산 아래 SVD 성분을 greedy하게 고른다. | **형식상 가장 가깝다.** Gram 가중 합 목적과 rank 예산이 이미 있다. 다만 실제 calibration Gram으로 WRRR를 풀지 않고, max 목적도 없다(서브에이전트 보고, 초록과 일치). |
| 10 | **Pareto Merging** (Chen & Kwok, ICML 2025, [2408.12105](https://arxiv.org/abs/2408.12105)), **MAP** (Li et al., ICLR 2025, [2406.07529](https://arxiv.org/abs/2406.07529)) | FULL_TEXT_READ_SUBAGENT | **Pareto Merging**: 선호 γ로 조건화한 smooth-Tchebycheff, 즉 부드러운 가중 max이며 data-free 변형은 Σ_t=I인 가중치 공간 이차식이다. rank 예산과 dual bound는 없다. **MAP**: 계수 공간에서 이차 대리모형을 세우고 NSGA-III로 푼다. | worst-task에 가까운 스칼라화는 이미 있다. |
| 11 | **METIS** (Im et al., 2026, [2606.16501](https://arxiv.org/abs/2606.16501)), **MergOPT** (Yang et al., ICLR 2026, [OpenReview](https://openreview.net/forum?id=C21rz8mo65)) | FULL_TEXT_READ_SUBAGENT / FULL_TEXT_UNVERIFIED | **METIS**: loss-gap softmax 과제 가중치(exp 가중)로 worst task 손실을 줄이는 정리를 제시하며 학습 데이터가 필요하다. **MergOPT**: fine-tuning 중에 가중치 공간 DRO를 쓴다. | 과제 가중치를 지수적으로 갱신하는 방식은 이미 있다. |
| 12 | **Cao et al.**, An Empirical Study and Theoretical Explanation on Task-Level Model-Merging Collapse. arXiv 2026-03. [2603.09463](https://arxiv.org/abs/2603.09463) | ABSTRACT_SELF + FULL_TEXT_READ_SUBAGENT | 서브에이전트 보고에 따르면 δ_max = max_i E‖h(X,θ̂)−h(X,θ_i)‖²로 정의하고 기하학적 하한 ¼Δ²를 준다(§3.4.1). rank 제약은 없다. | worst-task 기능 왜곡과 그 하한은 이미 다뤄졌다(기하 기반이며, 인스턴스별 Lagrange dual은 아님). |
| 13 | **Gao, Liu, Wang, Oh**, Rate Distortion for Model Compression. ICML 2019. [1810.06401](https://arxiv.org/abs/1810.06401) | FULL_TEXT_READ_SUBAGENT | 선형 모델에서 왜곡 E‖f_w−f_ŵ‖² = (w−ŵ)^TΣ(w−ŵ)이다. | 우리의 기능 오차 metric은 이미 알려진 왜곡 척도다. |

## 2. 수학적 토대 (이미 알려진 결과)

| 결과 | 출처 | 상태 | 이 프로젝트에서 쓰는 방식 |
|---|---|---|---|
| 출력 쪽 가중이 identity인 reduced-rank regression의 닫힌 해 | Izenman (1975), J. Multivariate Analysis 5(2):248–264 | FULL_TEXT_UNVERIFIED (메타데이터만 확인. 공식은 교과서 수준 결과이며 원시 데이터 reference로 수치 확인) | 고정 λ의 부분문제는 **고전 RRR 그 자체**다. 새 결과가 아니다. |
| 일반적인 원소별 가중 저계수 근사에는 닫힌 해가 없고 지역 최소가 있다 | Srebro & Jaakkola, ICML 2003 | FULL_TEXT_READ_SUBAGENT | 우리 부분문제는 한쪽 metric이라 여기에 해당하지 않는다. 출력 쪽에 과제별 가중을 두면 해당된다. |
| Fair PCA: 그룹별 최대 손실의 최소화, SDP 완화, 두 그룹이면 rank d+1, MW 해법(App. A) | Samadi et al., NeurIPS 2018, [1811.00103](https://arxiv.org/abs/1811.00103) | FULL_TEXT_READ_SUBAGENT | **task 가중치 mirror ascent + 닫힌 해 inner step 구조는 이미 알려져 있다.** |
| 두 그룹이면 SDP가 exact(Thm 1.2, 아핀 f_i도 포함, §2). k=3이면 gap 예시(Lemma 6.2: 7/4 vs 26/17). k 그룹에 추가 rank ⌊√(2k+1/4)−3/2⌋. k가 입력일 때 d=1에서 NP-hard | Tantipongpipat et al., NeurIPS 2019, [1902.11281](https://arxiv.org/abs/1902.11281) | **FULL_TEXT_READ_SELF** (Thm 1.2/1.3, §2 가정, Lemma 6.2) + 나머지는 SUBAGENT | D_t=I일 때 우리 문제는 min-max reconstruction fair PCA와 **정확히 같다**. 우리 dual은 Fantope/SDP 완화와 같다. Lemma 6.2 인스턴스를 테스트로 재현했다(LB=4−7/4, UB=4−26/17). |
| Socially fair low-rank approximation: 어떤 상수배 근사도 NP-hard | Song, Vakilian, Woodruff, Zhou, NeurIPS 2024, [2412.06063](https://arxiv.org/abs/2412.06063) | FULL_TEXT_READ_SUBAGENT | 전역 최적 solver를 기대하거나 주장하지 않는 근거다. |
| Worst-case PCA / StablePCA(Fantope 완화, 지수 가중 mirror-prox, 데이터 의존 certificate) | Fries et al. 2026 [2603.11304](https://arxiv.org/abs/2603.11304); Wang et al. [2505.00940](https://arxiv.org/abs/2505.00940) | FULL_TEXT_READ_SUBAGENT | dual/certificate 구조는 이미 알려져 있다. 둘 다 과제별 target이 없는 PCA형이다. |
| Group DRO의 지수 가중 갱신 | Sagawa et al., ICLR 2020, [1911.08731](https://arxiv.org/abs/1911.08731) | FULL_TEXT_READ_SUBAGENT | 같은 형태의 갱신 규칙이다. |

## 3. 이 프로젝트의 주장별 상태

| 주장 | 상태 | 근거 |
|---|---|---|
| A1. 고정 λ의 rank-k 가중 목적은 백색화 metric에서 Eckart–Young로 전역 최적해를 얻는다 | **이미 알려짐** (Izenman 1975의 특수 사례) | 원시 데이터 Izenman 공식, grid, ALS reference와 수치로 일치 |
| A2. full rank이면 λ-가중 RegMean과 같다 | **이미 알려짐** | 수치로 확인 |
| A3. 여러 과제 LoRA 병합에서 RegMean 해를 S^{1/2} metric으로 절단(joint WRRR)하는 것 | 확인한 출처에서는 찾지 못함. 가장 가까운 것은 Budget-Aware(proxy Gram + greedy), LoRM(factor 제한), ESM(과제별) | 고전 RRR을 직접 적용한 것에 가깝다. **기여로 주장하지 않는다.** |
| A4. max_t 목적의 Lagrange dual을 exponentiated-gradient로 풀고 inner를 닫힌 해로 계산 | **이미 알려진 알고리즘 구조** (Samadi et al. App. A, StablePCA, Group DRO) | – |
| A5. D_t=I 특수 사례의 dual gap과 계산 난이도 | **이미 알려짐** (Tantipongpipat 2019, Song 2024) | Lemma 6.2 재현 |
| A6. 과제별 target D_t와 과제별 공분산 Σ_t를 갖는 minimax RRR, 병합 모델에 대한 인스턴스별 dual 하한 | 확인한 출처에서는 찾지 못함(서브에이전트 두 명 모두 NOT_FOUND) | fair PCA를 target이 있는 경우로 일반화한 것에 가깝다. 신규성은 **미확인**이며 선언하지 않는다. |
| A7. worst-task 층별 기능 오차 최적화가 실제 과제의 worst-task 성능을 개선한다 | **미평가** (SCIENCE_NOT_EVALUATED) | 실제 adapter 파일럿이 BLOCKED 상태 |
