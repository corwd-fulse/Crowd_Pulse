#!/usr/bin/env bash
# 대시보드로 업로드된 영상을 원격 GPU worker 서버로 자동 복사한다.
#
# 이 PC(프론트·백엔드·DB)와 GPU 서버는 디스크를 공유하지 않는다. GPU worker는 DB의
# storage_path(예: uploads/videos/<id>.mp4)를 자기 디스크에서 찾으므로, 분석 요청 전에
# 같은 상대경로로 파일이 있어야 한다. 이 스크립트는 DB의 videos 행을 주기적으로 보고,
# GPU 서버에 없거나 크기가 다른 파일만 복사한다.
#
# - videos 행은 업로드 파일을 끝까지 쓴 뒤 commit되므로, 행이 보이면 로컬 파일은 완성본이다
#   (그래도 로컬 크기 == DB file_size를 한 번 더 확인한다).
# - 복사는 <파일>.part로 받은 뒤 sha256이 로컬과 같을 때만 원래 이름으로 바꾼다. 그래서
#   worker가 반쯤 받은 파일을 읽는 일이 없다.
# - 복사 완료한 경로는 상태 파일에 남겨 다음 확인부터 SSH 호출을 생략한다.
#
# 사용 (repo 루트에서, docker compose의 db 컨테이너가 떠 있어야 함):
#   GPU_HOST=gpu16 GPU_BACKEND_DIR=crowd-pulse/backend backend/scripts/sync_uploads_to_gpu.sh
# 환경변수:
#   GPU_HOST         ssh 설정(~/.ssh/config)의 호스트 이름 (기본 gpu16)
#   GPU_BACKEND_DIR  GPU 서버의 backend 디렉터리, 원격 홈 기준 (기본 crowd-pulse/backend)
#   POLL_SECONDS     DB 확인 주기 (기본 10)
#   STATE_FILE       복사 완료 목록 (기본 backend/uploads/.gpu_synced)
#   ONCE=1           한 번만 확인하고 종료
set -u

GPU_HOST=${GPU_HOST:-gpu16}
GPU_BACKEND_DIR=${GPU_BACKEND_DIR:-crowd-pulse/backend}  # 원격 홈 기준 상대경로(scp/ssh 둘 다 동작)
POLL_SECONDS=${POLL_SECONDS:-10}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BACKEND="$REPO_ROOT/backend"
STATE_FILE=${STATE_FILE:-"$BACKEND/uploads/.gpu_synced"}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)

touch "$STATE_FILE"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

list_videos() {
    # storage_path|file_size, 오래된 것부터
    (cd "$REPO_ROOT" && docker compose exec -T db sh -c \
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "select storage_path || '"'"'|'"'"' || file_size from videos order by uploaded_at"' \
        2>/dev/null) | tr -d '\r'
}

sync_one() {
    local path=$1 size=$2 local_file="$BACKEND/$1"
    # 경로 검증: uploads/videos/ 밑의 단순 파일명만 허용(원격 셸 명령에 그대로 들어가므로).
    if [[ ! $path =~ ^uploads/videos/[A-Za-z0-9._-]+$ ]]; then
        log "SKIP unexpected path: $path"; return 1
    fi
    if [[ ! -f $local_file ]]; then
        log "SKIP missing locally: $path"; return 1
    fi
    local local_size
    local_size=$(stat -c %s "$local_file")
    if [[ $local_size != "$size" ]]; then
        log "SKIP size mismatch (local=$local_size db=$size): $path"; return 1
    fi

    local remote_size
    remote_size=$(ssh "${SSH_OPTS[@]}" "$GPU_HOST" "stat -c %s $GPU_BACKEND_DIR/$path 2>/dev/null || echo none")
    if [[ $remote_size == "$size" ]]; then
        echo "$path" >> "$STATE_FILE"; log "ALREADY on GPU: $path"; return 0
    fi

    log "COPY $path ($size bytes)"
    local started=$SECONDS
    ssh "${SSH_OPTS[@]}" "$GPU_HOST" "mkdir -p $GPU_BACKEND_DIR/uploads/videos" || return 1
    if ! scp -q "${SSH_OPTS[@]}" "$local_file" "$GPU_HOST:$GPU_BACKEND_DIR/$path.part"; then
        log "FAIL scp: $path"; return 1
    fi
    local local_sha remote_sha
    local_sha=$(sha256sum "$local_file" | cut -c1-64)
    remote_sha=$(ssh "${SSH_OPTS[@]}" "$GPU_HOST" "sha256sum $GPU_BACKEND_DIR/$path.part | cut -c1-64")
    if [[ $local_sha != "$remote_sha" ]]; then
        log "FAIL sha256 mismatch, removing partial: $path"
        ssh "${SSH_OPTS[@]}" "$GPU_HOST" "rm -f $GPU_BACKEND_DIR/$path.part"
        return 1
    fi
    ssh "${SSH_OPTS[@]}" "$GPU_HOST" "mv $GPU_BACKEND_DIR/$path.part $GPU_BACKEND_DIR/$path" || return 1
    echo "$path" >> "$STATE_FILE"
    local secs=$((SECONDS - started)); ((secs == 0)) && secs=1
    log "DONE $path in ${secs}s ($((size / secs / 1024 / 1024)) MiB/s, sha256 ok)"
}

log "watching videos table; GPU_HOST=$GPU_HOST dir=$GPU_BACKEND_DIR every ${POLL_SECONDS}s"
while true; do
    while IFS='|' read -r path size; do
        [[ -z $path ]] && continue
        grep -qxF "$path" "$STATE_FILE" && continue
        sync_one "$path" "$size"
    done < <(list_videos)
    [[ ${ONCE:-0} == 1 ]] && break
    sleep "$POLL_SECONDS"
done
