# Baseline Report

This stage completed a remote-only MMAction2 pretrained medium baseline for traffic accident video recognition.

- Data: ACCIDENT real, medium profile train=300/val=112/test=300.
- Models: VideoMAE-B, SlowFast-R50, X3D-S using Kinetics pretrained checkpoints cached on the remote server.
- Outputs: predictions, metrics, threshold sweeps, checkpoints, and error-case grids under `experiments/accident_mmaction2_baseline/outputs/`.
- Main caveat: `non_accident` remains a weak pre-event clip label; no true hard-negative class is trained or reported.

See `model_comparison_report.md`, model-specific reports, `error_analysis.md`, and `next_stage_plan.md`.
