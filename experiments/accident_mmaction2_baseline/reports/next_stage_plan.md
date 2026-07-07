# Next Stage Plan

1. Download/cache official pretrained checkpoints for VideoMAE, SlowFast-R50 and X3D-S/M under `models/pretrained/`, then update configs with `load_from`.
2. Expand annotations beyond the 40/40/40 smoke subset and preserve source video/event/camera grouping.
3. Add real normal and hard-negative CCTV clips; do not label YOLO boxes as accident labels.
4. Run class-balanced fine-tuning with threshold calibration and report accident recall, precision, PR-AUC, false alarms, latency and GPU memory.
5. Add YOLO/Track evidence overlay and candidate trigger features for speed drop, abnormal stop, trajectory conflict, overlap and queue growth.
6. Defer ONNX/TensorRT/service hardening until the model choice is defensible.
