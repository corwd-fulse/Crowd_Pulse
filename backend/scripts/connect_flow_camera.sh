#!/usr/bin/env bash
# 분석 job 하나를 '축제 전체 흐름' 지도의 카메라로 연결한다(분석이 끝날 때까지 기다렸다가 자동 실행).
#   1) job이 succeeded 될 때까지 대기(failed면 중단)
#   2) GPU 서버에서 경계선 통과 수 계산(python -m app.gate_flow) -> 이 PC의 uploads/results/<job>/gate_flow.json 로 복사
#   3) configs/festival_flow.json 의 카메라 job_id 연결
#   4) 행사장 지도(venue) 구역의 영상 ROI 매핑을 새 버전으로 저장
# usage: connect_flow_camera.sh <job_id> <camera_id: cam-top|cam-mid|cam-bottom> <axis: x|y> <venue_zone_id>
set -euo pipefail
JOB=$1 CAM=$2 AXIS=$3 ZONE=$4
REPO=/c/Users/Woori/crowd_pulse
PY="$REPO/.venv-forecast/Scripts/python.exe"
API=http://localhost:8000
log() { echo "$(date '+%F %T') $*"; }
q() { docker exec crowd-pulse-db sh -c "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -At -c \"$1\""; }

while true; do
  st=$(q "select status from analysis_jobs where id='$JOB'")
  case "$st" in
    succeeded) log "job $JOB succeeded"; break;;
    failed) log "job $JOB failed: $(q "select error_message from analysis_jobs where id='$JOB'")"; exit 1;;
    *) sleep 60;;
  esac
done

OUT=$(ssh -o BatchMode=yes gpu16 "ls -d ~/crowd-pulse/backend/uploads/analysis/$JOB/*/ | head -1")
log "gate counts on gpu16 from $OUT"
ssh -o BatchMode=yes gpu16 "cd ~/crowd-pulse/backend && venv/bin/python -m app.gate_flow $JOB $OUT --axis $AXIS"
mkdir -p "$REPO/backend/uploads/results/$JOB"
scp -q -o BatchMode=yes "gpu16:crowd-pulse/backend/uploads/results/$JOB/gate_flow.json" "$REPO/backend/uploads/results/$JOB/gate_flow.json"

ROI=$(q "select roi_snapshot->'zones'->0->>'name' from vision_results where analysis_job_id='$JOB'")
log "connect $CAM -> $JOB (ROI $ROI), venue zone $ZONE"
"$PY" - "$REPO/backend/configs/festival_flow.json" "$CAM" "$JOB" "$ZONE" "$ROI" "$API" <<'PYEOF'
import json, sys, urllib.request
path, cam, job, zone, roi, api = sys.argv[1:]
cfg = json.load(open(path, encoding="utf-8"))
for c in cfg["cameras"]:
    if c["id"] == cam:
        c["job_id"] = job
        c["meaning"] = c["meaning"].replace(" (분석 끝나면 연결)", "")
json.dump(cfg, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
v = json.load(urllib.request.urlopen(f"{api}/api/venues/{cfg['venue_id']}"))
conf = v["configuration"]
for k in ("mapping_roi_snapshots", "camera_sources"):
    conf.pop(k, None)
for z in conf["zones"]:
    if z["id"] == zone:
        z["mapping"] = {"job_id": job, "roi_name": roi}
        z["note"] = f"{cam} 카메라 영상(job {job[:8]}) 연결"
req = urllib.request.Request(f"{api}/api/venues/{cfg['venue_id']}/revisions",
                             data=json.dumps({"base_revision": v["revision"], "configuration": conf}).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
print("venue revision", json.load(urllib.request.urlopen(req))["revision"])
flow = json.load(urllib.request.urlopen(f"{api}/api/venues/{cfg['venue_id']}/festival-flow"))
for c in flow["cameras"]:
    g = c["gate"]
    print(c["id"], c["job_id"], g and f"{g['seconds']}s plus {sum(g['plus'])} minus {sum(g['minus'])}")
PYEOF
log "done"
