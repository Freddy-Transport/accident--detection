# Full Profile Data Report

- Clip root: `/autodl-fs/data/traffic_accident_rnd/ACCIDENT_mmaction_clips/full`
- Group leakage count: `0`

- train: written `625`, labels `{'non_accident': 219, 'accident': 406}`, decode checked `30`, decode failed `0`
- val: written `153`, labels `{'accident': 101, 'non_accident': 52}`, decode checked `30`, decode failed `0`
- test: written `2468`, labels `{'non_accident': 948, 'accident': 1520}`, decode checked `30`, decode failed `0`

Note: `non_accident` clips are pre-event windows from ACCIDENT videos, not real hard negative CCTV samples.
