# GRAPH-TRAVERSE

TRM 기반 그래프 경로 추론 파이프라인입니다.

## Quick Start

1. 의존성 설치
```bash
pip install -r requirements.txt
```

2. TinyRecursiveModels 준비
```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git TinyRecursiveModels
```

3. 데이터 배치
- `data/webqsp/*` 또는 `data/CWQ/*` 경로에 데이터 파일을 둡니다.
- 기본 경로는 `trm_rag_style/configs/*.json`에 정의되어 있으며 모두 저장소 루트 기준 상대경로입니다.

4. 실행
```bash
DATASET=webqsp bash trm_rag_style/scripts/run_preprocess.sh
DATASET=webqsp bash trm_rag_style/scripts/run_embed.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_rag_style/scripts/run_train.sh
```

## Notes
- 기존 엔트리포인트 `python -m trm_gnnrag_style.run`는 호환 alias로 유지됩니다.
- `TRM_ROOT` 환경변수로 TinyRecursiveModels 경로를 재지정할 수 있습니다.
