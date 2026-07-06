# 运行手册

所有命令在远程服务器执行：

```bash
cd /root/autodl-tmp/traffic_accident_rnd
source .venv/bin/activate
```

## 环境检查

```bash
scripts/env_check.sh
```

## 数据检查

```bash
python scripts/check_kaggle_dataset.py --dataset accidentbench/accident-benchmark --max-gb 5
python scripts/validate_manifest.py data/manifests/example_manifest.jsonl
```

没有 `/root/.kaggle/kaggle.json` 时，Kaggle 检查只记录 `blocked_no_token`，不会下载数据。

## 生成 smoke 视频

```bash
python scripts/generate_smoke_video.py --output data/samples/smoke_accident_like.mp4
```

## 候选片段触发

```bash
python scripts/run_trigger.py --video data/samples/smoke_accident_like.mp4 --output outputs/candidates/smoke_candidates.jsonl
```

## 模型 smoke test

```bash
python scripts/train_baseline.py --smoke --output models/checkpoints/smoke_r3d18.pt
python scripts/infer_video.py --video data/samples/smoke_accident_like.mp4 --output outputs/inference/smoke_prediction.json
```

## 推理服务

```bash
uvicorn traffic_accident_rnd.api:app --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict/video \
  -H 'Content-Type: application/json' \
  -d '{"video_path":"/root/autodl-tmp/traffic_accident_rnd/data/samples/smoke_accident_like.mp4"}'
```
