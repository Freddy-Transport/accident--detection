# Sanity Check Report

- Video decode check: 40/40 sampled clips readable. Log: `reports/check_videos.log`.
- Native tiny GPU sanity: 10 batches, tensor `[1, 3, 16, 224, 224]`, logits `[1, 2]`, finite loss, peak GPU memory about 175 MB. Log: `reports/videomae_native_sanity.log`.
- MMAction2 import sanity: `mmaction_import=ok`, one batch finite loss, peak GPU memory about 175 MB. Log: `reports/mmaction_import_sanity.log`.
- Full MMAction2 smoke: VideoMAE/SlowFast/X3D each trained for 1 epoch on 40 train clips and evaluated on 40 test clips.

These are smoke results only; the negative class is derived from pre-event windows.
