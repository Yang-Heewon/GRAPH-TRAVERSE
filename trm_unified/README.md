# trm_unified

CWQ/WebQSP 공통 TRM 파이프라인입니다.

## 실행 전 준비
1. 의존성 설치
```bash
pip install -r requirements.txt
```
2. 데이터 준비
- WebQSP: `data/webqsp/{train.json,dev.json,entities.txt,relations.txt}`
- CWQ: `data/CWQ/...`

## 스크립트 실행 (저장소 어디에 clone해도 동작)
```bash
DATASET=webqsp bash trm_unified/run_preprocess.sh
DATASET=webqsp bash trm_unified/run_embed.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_unified/run_train.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_unified/run_test.sh
```

## 직접 CLI 실행
```bash
python -m trm_unified.pipeline --help
```

- `--trm_root` 기본값은 `<repo>/TinyRecursiveModels`입니다.
- 필요 시 `TRM_ROOT` 환경변수 또는 `--trm_root`로 변경하세요.
- 기본적으로 저장소에 포함된 `TinyRecursiveModels`(로컬 사용본)를 그대로 사용합니다.
