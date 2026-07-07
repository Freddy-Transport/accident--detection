# Next Stage Plan

1. Add true normal and hard-negative fixed-camera CCTV clips before increasing training epochs.
2. Fix SlowFast config/checkpoint alignment or switch to the exact dumped MIM SlowFast config before drawing model-quality conclusions.
3. Run VideoMAE-B 3-10 epoch fine-tuning with validation-threshold selection and class-balanced sampling after hard negatives are available.
4. Keep X3D-S as the low-latency deployment candidate and retest after hard negatives are added.
5. Add YOLO/Track evidence overlays and candidate trigger features for speed drop, abnormal stop, trajectory conflict, bbox overlap and queue growth.
6. Defer ONNX/TensorRT and service hardening until a threshold with acceptable recall and false alarm rate is validated on real hard negatives.
