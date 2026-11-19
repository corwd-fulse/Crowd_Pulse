# User-defined zones and observed 2D replay

Apply additive Alembic revision `20261113_06` before running the updated API:

```powershell
cd backend
.venv\Scripts\python.exe -m alembic upgrade head
```

This creates `video_roi_revisions` only; no prior video/job/result/CSV is
rewritten. The local workspace DB has already been upgraded. No package changes
are needed. CPU `full_frame` Optical Flow and its confidence policy stay default.
The existing Crop128 option remains GPU TEST-only, as documented in HYBRID_ANALYSIS.md.

## Zone editing and snapshots

The video page's existing polygon editor now defaults to A/B/C…Z/AA… and supports
name, color, display visibility, vertex drag and numeric original-pixel edits.
Polygons must be nonempty, non-self-intersecting, disjoint and non-touching;
unassigned gaps are outside. Hidden zones still participate in analysis.
New versions are candidate/unapproved; editing never silently retains approval.

`GET /api/videos/{id}/roi` returns the effective snapshot, ordinal and fingerprint.
`POST /api/videos/{id}/roi-revisions` requires `If-Match: <fingerprint>`, validates
actual source resolution and geometry, locks the video to serialize edits, and
appends a new snapshot. Stale edits get 412; missing precondition gets 428.
Original `videos.roi_snapshot` is the immutable legacy fallback.

The initial-registration PUT contract is retained. Ordinary PUT replacement of
an existing ROI is rejected. The existing explicit approval PUT is retained
**only for identical zones/resolution** and appends an approved copy, never
overwriting the candidate. User editor saves never invoke approval. Job creation
copies the latest committed snapshot; later names/coordinates/colors/revisions
cannot affect requested or completed jobs. `revision_id` travels inside snapshots.

## Observed transitions and OD

`zone_replay.py` uses original image bbox bottom-centers. Defaults: 3 consecutive
frames to confirm, 3px signed-distance hysteresis, 15px net displacement to count
a change. Initial dwell establishes state without inventing an arrival. Suspected
ID jumps (>40px and >6x recent median step), missing frames/timestamps and overlaps
split history; segments are never reidentified or linked across unknown gaps.

Each event includes stable event ID, track/segment ID, from/to, frame, relative
timestamp_ms, observed positions and heuristic confidence. Confidence is detection
support, not calibrated probability that the identity or direction is correct.

`kind=boundary` alone updates zone inflow/outflow. Observed A→outside→B produces
two boundary events and one `kind=zone_path` route event; this avoids double
counting zone flow while preserving A→B route counts. Direct A→B is one boundary
event and one route count. Reentries are distinct events. A→B→A counts two route
transitions but one observed OD A→A; A→B→C has OD A→C. Fragmented IDs are excluded
from OD. All OD endpoints are observation-censored, not actual journey origins or
destinations or unique-person counts across ID changes.

Missing paths make cumulative total inflow/outflow/net NULL. Confirmed event
counts are retained separately as lower bounds. Occupancy is observed detections,
not calibrated true population. A missing frame index is unknown, never a zero
person count. A verified processed empty frame can legitimately have zero boxes.

## Storage and read-only replay

New FAST/FULL pipelines add `zone_transitions.json` and include its policy,
events/routes/OD/diagnostics in `execution_snapshot.settings.transitions`. The five
schema-v1 CSVs remain unchanged. The Worker hashes the new computation sources.

`GET /api/analysis/jobs/{id}/replay` validates worker ownership and matching
execution metadata through the existing detection adapter, and validates the
tracker frame-index count/timestamps against detection rows. Completed historical
jobs derive events read-only from immutable CSVs and their **analysis snapshot**;
current edited video zones are never substituted. New recorded events must match
the replay version/computation or the endpoint rejects reinterpretation.
The response exposes event provenance and policy version. A bounded 4-entry cache
is invalidated by source file stats and snapshot changes. Existing CSV size/frame
limits remain; this is not a streaming API for arbitrary-length footage.

## Dashboard

The video and 2D canvas share the actual HTML video clock, source FPS and observed
frame numbers. Seeking, pause/play and .25/.5/1/2/4x speed all control that same
video. Unknown/missing frames show no marker; trajectories are not interpolated.
Canvas is an image-coordinate replay, not a calibrated top-down floor map or a
future simulation. Marker/event path filtering uses confirmed recorded routes;
zone totals remain all-observation context and are labeled accordingly.

The existing count chart/result tables remain. Pixel coordinates, current box
counts, per-second stored means, confirmed transitions and observed OD are distinct
quantities. Risk is shown only for an observed frame with available stored scores
and no known missing current/previous window metrics; unknown remains unanalysed.
No physical density/congestion percentage is invented without area calibration.

Verification and performance: `../validation/zone_replay_20261113/REPORT.md`.
