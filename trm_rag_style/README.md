# trm_rag_style

TRM-RAG 실행 오케스트레이터입니다.

## 전제
- 저장소 루트에서 실행
- 기본 TRM 모듈은 저장소의 `TinyRecursiveModels`(로컬 사용본)
- 필요 시 `TRM_ROOT` 환경변수로 다른 경로 지정 가능
- 데이터 파일이 `data/` 아래 존재

## 기본 실행
```bash
DATASET=webqsp bash trm_rag_style/scripts/run_preprocess.sh
DATASET=webqsp bash trm_rag_style/scripts/run_embed.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_rag_style/scripts/run_train.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_rag_style/scripts/run_test.sh
```

## 단일 엔트리포인트
```bash
python -m trm_rag_style.run --dataset webqsp --stage all --model_impl trm_hier6
```

## 경로 규칙
- `configs/base.json`, `configs/webqsp.json`, `configs/cwq.json`의 경로는 저장소 루트 기준 상대경로입니다.
- 필요 시 `--override`로 덮어쓸 수 있습니다.
```bash
python -m trm_rag_style.run --dataset webqsp --stage train --override trm_root=/abs/path/to/TinyRecursiveModels
```

## 호환성
- `trm_gnnrag_style` 패키지는 `trm_rag_style`로 포워딩되는 호환 alias입니다.
