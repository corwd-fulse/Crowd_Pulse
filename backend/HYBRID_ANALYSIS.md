# Experimental ROI Crop hybrid analysis

Default Worker execution remains `--device cpu --imgsz 1920 --flow-method full_frame`.
GPU/model selection is explicit; there is no automatic device fallback or model download.
No database migration is required. Historical jobs, results and CSV files are immutable.

For a **new, queued TEST job** with FULL profile, use the existing GPU environment:

```powershell
cd C:\Users\Woori\Desktop\Crowd_Pulse\backend
.\out\gpu-validation-20261112\.venv-gpu\Scripts\python.exe -m app.worker `
  --model-path .\out\gpu-validation-20261112\yolo11n.pt `
  --algorithm-version hybrid-roi-crop128-v1 `
  --device cuda:0 --imgsz 960 --flow-method roi_crop_128 `
  --allow-test-analysis --once --job-id "<new test job UUID>"
```

Do not interrupt an existing Worker to run this command. Existing local/DB locks
remain enforced. Crop rejects CPU execution, unscoped queues, and production jobs.
FAST remains tracking/observations/individual directions only even when the Worker
requested crop: metadata records `flow_method=not_run` and the requested method.
Selecting FULL creates a separate new job; it never upgrades/overwrites FAST output.

## Flow and tracking semantics

Crop uses the union bounding rectangle of configured polygons with 128 original
image pixels of padding, clipped to video bounds. Farneback parameters remain
`0.5, 3, 15, 3, 5, 1.2, 0`. ROI masks use the crop offset; displacement vectors do
not need a translation because both frames share the same crop. Full-image
scene-change checks, original-image diagonal normalization, per-second vector
aggregation, and `flow.csv` fields remain unchanged. Empty/invalid support is
low_confidence with missing values, not zero. Scene-change confidence is not
silently relaxed to obtain more scores.

ByteTrack individual directions remain independent and use bbox bottom centers.
Nonconsecutive observations/nonincreasing timestamps and suspected ID jumps are
unreliable with blank direction/displacement. `speed_px` retains its historical
meaning: displacement over the observation window, **not px/frame or m/s**.
No ID merging or automatic re-identification was added.

`hybrid_flow.py` adds one circular direction vote per valid ID per second. Each
person's circular coherence must be >=0.3 and displacement >=1 pixel. Comparisons
require >=3 distinct valid IDs, aggregate coherence >=0.3, available Dense Flow,
and Dense net magnitude >=0.25px/frame. Circular difference <=45° is agreement;
otherwise the independent numerical results are preserved and hybrid confidence
is low. These are experimental, uncalibrated thresholds. Flow/person magnitudes
are never combined. Individual speed in the additive summary uses actual elapsed
timestamps in px/s; directional counts are distinct valid IDs per second, not
unique visitors or physical density.

## Storage/API compatibility and confidence

The five CSV schema-v1 headers and result manifest remain unchanged. Additional
data is written to the new run's `pipeline/analysis_details.json`, then included
in `execution_snapshot.settings`: actual/requested Flow method, padding,
Farneback parameters, timings, score formula/missing policy, hybrid version,
thresholds, per-zone/second votes, speed, direction distribution, comparison and
confidence reasons. Existing Worker metadata also records model/source hashes,
verified device, imgsz, tracker, precision, torch/CUDA versions and frame metadata.

GET result summary adds `bottleneck_status`. Zone metrics add nullable `hybrid`.
Historical results return no hybrid data; their stored values are never rewritten.
The React dashboard separates area Flow, historical individual sample averages,
and distinct-person hybrid estimates. Agreement never upgrades the independent
global direction-quality/ROI review status. Existing video/boxes/ROI remain.

## Bottleneck score policy

Valid calculations retain the existing weights (0.40 growth, 0.35 slowdown,
0.25 imbalance), capacities and 40/70 risk thresholds. The new pipeline uses
`bottleneck_missing_policy=omit_unavailable_v1`: any missing count/motion/inflow/
outflow in a zone window prevents its score. A missing previous window also
prevents the dependent slowdown comparison. Missing scores are omitted from the
nonnullable legacy CSV/DB score table and explicitly listed as `unanalyzed` in
metadata/API/UI. Component status can be `partial` or `unanalyzed`; FAST stays
`not_run`. Missing does not mean low risk or zero. The first available initial
window retains the legacy growth/slowdown initialization convention; window size
and handling of a partial final window are unchanged. Formula and missing policy
are separately versioned so past/new outputs can be interpreted correctly.

The low-level score function retains its explicit `legacy` default for historical
reproduction; the new pipeline always requests the stricter missing policy.

## Verification

See `../validation/hybrid_system/REPORT.md`. One real full GPU test job was executed
after unit/integration/frontend and a 21-frame real-video regression passed.
No automatic full-video retry is permitted by the validation runner's launch guard.
