# PAPER_STATUS: Calibration Sensitivity in Minimax Low-Rank Approximation

| 항목 | 상태 |
|---|---|
| 버전 | Working Draft **v0**. 내부 초안이며 제출하지 않았다. v0 텍스트는 커밋 `2e6418e`에 저장했고, 그 뒤 stage-3 결과를 반영했다 |
| 형식 | 공식 ICLR 2027 style ZIP을 수정 없이 사용했다(sha256 `0d940dfa…`, 파일별 hash는 `STYLE_SHA256.txt`). anonymous 모드이며 running header만 "Internal working draft v0 — not submitted"로 바꿨다(`BUILD.md` 참고) |
| 빌드 | **BUILT**: `paper/main.pdf`. pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian) |
| 쪽수 | 전체 16쪽. **본문은 1–9쪽**이며 Conclusion이 9쪽에서 끝난다(`build_report.json`의 main_text_ends_on_page=9). 이어서 AI use/Reproducibility statement, 참고문헌, 부록이 온다 |
| 경고 | undefined reference 0, undefined citation 0, overfull hbox 0, LaTeX warning 0, BibTeX warning 0. underfull hbox는 8개로 줄바꿈 미관 문제다 |
| 렌더링 확인 | `pdftoppm`으로 1, 5, 7, 8, 9, 10, 13, 14쪽을 렌더링해 표, 수식, 그림, 참고문헌, running header를 눈으로 확인했다 |
| 빌드 CPU | build와 render 합계 14.9초 / 상한 600초(`build_log.jsonl`) |
| 분석 CPU | stage-3 bound check 216.1초(상한 1,100초 RLIMIT). exporter와 bib 생성은 실행당 약 0.1초이며 개별 합산은 하지 않았다. 과제 상한 1,200초 안이다 |
| 설치 비용 | apt 1차 시도 실패(3.7 CPU-s), 2차 성공(52.2 CPU-s, 52.3초 wall, 다운로드 69.1 MB). 분석과 빌드 예산에는 포함하지 않았다 |
| 사람 검토 | **HUMAN_REVIEW_PENDING** |
| 과학 | SCIENCE_NOT_EVALUATED. 실제 adapter는 BLOCKED이고 원인 판정은 CAUSE_UNDETERMINED다 |

## 섹션별 완성도

| 섹션 | 상태 |
|---|---|
| Abstract, Introduction, Setting, Experiments, Discussion/Limitations, Conclusion | 영어 문장으로 완성했다. 모든 수치는 exporter 매크로(`generated/numbers.tex`, 155개)에서 온다 |
| Related Work | 완성했다. 대부분의 특성 서술은 서브에이전트 요약에 근거한다(`claim_evidence.tsv` C35) |
| Analysis | Proposition 2개, Lemma 3개, Corollary 1개. 모두 표준 결과이며 증명은 AI가 작성했고 사람 검토 전이다 |
| Appendix | 증명, 구현 감사, 추가 표, bound 수치 점검, 정정 이력, Planned (NOT RUN), 재현 정보 |

## 실행한 것과 실행하지 않은 것

- 실행한 것
  - stage-3 bound check: 24/24 cell OK. 행렬 27개를 재생성했고 hash 27/27이 일치했다.
- 실행하지 않은 것 (원고 Appendix E에 모두 적었다)
  - 실제 adapter 파일럿: BLOCKED
  - 절대오차와 상대오차 비교
  - n_cal 의존성 sweep: 승인되지 않았다
  - 정규화된 worst-task 변형: 구현하지 않았다

## 숫자와 인용

- 수치는 `tools/export_results.py`가 raw result에서 생성한다. 출처 추적은 `generated/provenance.tsv`에 있다.
- 인용은 `tools/make_bib.py`가 fetched metadata에서 생성한다. 원본 metadata는 `bib_sources/`에 있다. venue는 arXiv comment에 적힌 경우에만 표기한다.
- 주장별 근거는 `claim_evidence.tsv`(41행)에 있다.
- 요약과 원자료가 달랐던 부분은 `../CORRECTIONS.md`와 원고 Appendix F에 기록했다.
