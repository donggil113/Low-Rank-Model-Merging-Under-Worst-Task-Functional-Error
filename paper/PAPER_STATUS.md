# PAPER_STATUS: Calibration Sensitivity in Minimax Low-Rank Approximation

| 항목 | 상태 |
|---|---|
| 버전 | Working Draft **v0**. 내부 초안이며 제출하지 않았다 |
| 형식 | 공식 ICLR 2027 style ZIP 사용(sha256 `0d940dfa…`). anonymous 모드이고, running header만 "working draft, not submitted"로 바꿨다(`BUILD.md` 참고) |
| 빌드 | **NOT_BUILT (v0 저장 시점)**. 이 환경에는 LaTeX 엔진이 없다. 빌드는 다음 단계에서 진행한다 |
| 사람 검토 | **HUMAN_REVIEW_PENDING** |
| 과학 | SCIENCE_NOT_EVALUATED. 실제 adapter는 평가하지 않았다(BLOCKED). 원인 판정은 CAUSE_UNDETERMINED |

## 섹션별 완성도 (v0)

| 섹션 | 상태 |
|---|---|
| Abstract, Introduction, Setting, Experiments, Discussion/Limitations, Conclusion | 영어 문장으로 완성했다. 모든 수치는 exporter 매크로에서 온다 |
| Related Work | 완성했다. 대부분의 문헌 특성 서술은 서브에이전트 요약에 근거한다(`claim_evidence.tsv` C35) |
| Analysis | Proposition 2개, Lemma 3개, Corollary 1개. 모두 표준 결과이며, 증명은 AI가 작성했고 사람 검토 전이다 |
| Appendix | 증명, 구현 감사, 추가 표, 정정 이력, Planned (NOT RUN), 재현 정보 |
| 수식 bound의 수치 점검 | v0 시점에는 **NOT RUN**(Appendix E 5번). 다음 단계에서 수행한다 |

## 숫자와 인용

- 수치는 `tools/export_results.py`가 raw result에서 생성한다. 출처 추적 파일은 `generated/provenance.tsv`다.
- 인용은 `tools/make_bib.py`가 fetched metadata에서 생성한다. 원본 metadata는 `bib_sources/`에 있다.
- 주장별 근거는 `claim_evidence.tsv`에 있다.
