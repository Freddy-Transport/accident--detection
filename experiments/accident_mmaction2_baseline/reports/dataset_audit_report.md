# ACCIDENT Dataset Audit

- Dataset root: `/autodl-fs/data/traffic_accident_rnd/ACCIDENT_dataset`
- Metadata rows: `2027`
- Manifest rows: `3246`
- Video extensions: `{'.mp4': 2027}`
- Accident type counts: `{'rear-end': 328, 't-bone': 657, 'single': 680, 'head-on': 117, 'sideswipe': 245}`
- Split counts: `{'test': 1520, 'train': 507}`
- Explicit non-accident labels: `false`
- Explicit hard-negative labels: `false`
- Manifest is clip-level and grouped by `source_video_id` / `group_id`.

ACCIDENT real is accident-centric. Non-accident samples in this baseline are weak pre-event clips, not real normal traffic videos.
