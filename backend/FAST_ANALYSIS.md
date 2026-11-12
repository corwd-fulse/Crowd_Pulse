# FAST / FULL analysis

`analysis_profile` is independent of `analysis_mode` (production/test ROI permission).
`POST /api/analysis/jobs` accepts:

```json
{"video_id":"<video UUID>","analysis_mode":"test","analysis_profile":"fast"}
```

Omitting `analysis_profile` keeps the historical **full** backend default. React
request controls default to **fast** and also offer FULL. Candidate ROI still
requires test permission on both API and Worker. Approvals are never automatic.
The same per-video queued/running unique index applies across both profiles.

FAST computes YOLO/ByteTrack, observations, individual directions and zone
aggregates. It never invokes Optical Flow or bottleneck scoring. All five CSV
headers remain compatible; flow and bottleneck CSVs contain headers only.
Validation rejects nonempty flow/scores for a FAST import. Job and execution
profiles must match. FULL preserves the original pipeline and default signature.

Both profiles retain queued/running/succeeded/failed/cancelled status values.
Use **status + analysis_profile** to distinguish FAST completed from FULL
completed; succeeded FAST does not imply precise analysis completed.
Results summary additionally returns `components.optical_flow` and
`components.bottleneck` as `not_run` for FAST (`completed` for FULL). The scores
endpoint also returns `component_status`. Empty FULL scores can be a legitimate
short-video result and differ from FAST `not_run`.
FAST zone metrics keep `has_flow=false` and flow values / low_confidence NULL,
not zero or false. Individual reliability remains in raw CSV. Directions retain
the existing quality/ROI checks; FAST numerical estimates are explicitly marked
unvalidated until reviewed. They are never presented as confirmed directions.

Migration **20261112_05** only adds a constrained `analysis_jobs.analysis_profile`
column with default full; old results and states are unchanged. FULL historical
import fingerprint normalization omits the default new field to preserve retry
idempotency for legacy metadata. No previous migration is edited.

Worker CLI is unchanged: it reads the requested profile from the claimed job.
Example PowerShell GPU command (replace job UUID with a new queued job):

```powershell
cd C:\Users\Woori\Desktop\Crowd_Pulse\backend
.\out\gpu-validation-20261112\.venv-gpu\Scripts\python.exe -m app.worker `
  --model-path .\out\gpu-validation-20261112\yolo11n.pt `
  --algorithm-version vision-fast-gpu-v1 `
  --device cuda:0 --imgsz 960 --allow-test-analysis --once `
  --job-id "<new queued job UUID>"
```

The same command executes FULL when that job requested full. It never promotes
or overwrites FAST results. After FAST succeeds, select FULL on the video page
and create a **new** job. One active job per video remains enforced.

## Detection reuse assessment

FULL currently runs tracking again, preserving the existing execution contract.
Automatic reuse is deliberately not enabled: the source detection CSV is a
Worker output outside the five-file result manifest and needs a validated cache
contract first. Reuse must check source video SHA256, frame count/FPS/resolution,
model SHA256, imgsz, precision/device and tracker/config/code version, validate
CSV timestamps/schema/IDs, copy or reference immutable bytes with a manifest,
and preserve the source job and provenance. ROI need not match for tracking
reuse, but zone calculations must use the new job ROI. A completed FAST job alone
is not enough to prove cache compatibility. Skipping the expensive flow is the
main current gain; avoiding a tracking rerun is a separate enhancement.

## One actual GPU test

Artifacts: `out/fast-worker-e2e-20261112/`.
Video: `c795a974-f7d0-4981-a018-d35725c88833`.
Job: `396a1bcd-b9a3-4273-bd97-01d8590fe46f`.
Test/candidate/unvalidated preserved. One real tracking call, no automatic retry.
See that directory's report for measured timings and UI screenshots.
