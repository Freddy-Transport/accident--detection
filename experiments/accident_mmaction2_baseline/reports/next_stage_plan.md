# Next Stage Plan

## Recommended Next Steps

1. Add real CCTV hard negative data: congestion, bus stop, temporary parking, construction, night glare, rain/fog, occlusion, and camera shake.
2. Re-train/evaluate VideoMAE with `accident / normal / hard_negative` once labels are available; keep current binary model as baseline.
3. Run longer RTSP-style validation on fixed-camera streams and measure false alarms per camera-hour.
4. Calibrate candidate trigger thresholds separately per camera ROI and lane geometry; keep YOLO/Track as evidence, not accident truth.
5. Package the exported TorchScript or ONNX model behind the pipeline preprocessing wrapper before considering FastAPI/gRPC deployment.
6. Consider TensorRT only after ONNX preprocessing and numerical parity are stable on a representative validation set.

## Current Deployment Candidate

Use the full pipeline demo with `--tracker auto`, threshold `0.56`, and dry-run push first. Do not enable real alarms until hard negative CCTV false alarm testing is complete.
