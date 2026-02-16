# trm_rag_style (legacy)

이 경로는 레거시 호환용입니다. 신규 사용은 `trm_agent` 경로를 권장합니다.

## 전제
- 저장소 루트에서 실행
- 기본 TRM 모듈은 저장소의 `TinyRecursiveModels`
- 데이터 파일이 `data/` 아래 존재

## 데이터 준비 + 전처리
```bash
DATASET=webqsp bash scripts/setup_and_preprocess.sh
```

## 기본 실행
```bash
DATASET=webqsp bash trm_agent/scripts/run_embed.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_agent/scripts/run_train.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_agent/scripts/run_test.sh
```

## 단일 엔트리포인트
```bash
python -m trm_agent.run --dataset webqsp --stage all --model_impl trm_hier6
```
