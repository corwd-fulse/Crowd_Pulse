import {EXIT_SIDES, averageRates, cumulativeAt, exitShares, minutesToEmpty, zoneFlowAt} from '../../services/festivalFlow';
import type {FestivalFlowConfig, FlowTimeline} from '../../services/festivalFlow';

// 축제 전·후 비교 패널: 축제 전 -> 축제 중 인원 증가, 축제 후 유출 속도·방향·비워지는 시간.
// 카메라 분석(YOLO+ByteTrack) 경계선 통과 수만 쓴다. 예측은 '평균 속도 유지' 가정.

const f1 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(1));
const ratio = (a: number | null | undefined, b: number | null | undefined) => (a == null || b == null || b === 0 ? '—' : `${(a / b).toFixed(1)}배`);

export default function FestivalPhasePanel({cfg, timeline, initials, phase, zoneName, time = 0}: {
  cfg: FestivalFlowConfig; timeline: FlowTimeline; initials: Record<string, number>; phase: 'pre' | 'post'; zoneName: (id: string) => string; time?: number;
}) {
  const ph = cfg.phases?.[phase];
  if (!ph) return <p className="venue-muted">이 단계 설정이 없습니다.</p>;
  const during = (id: string) => cfg.cameras.find((c) => c.id === id) ?? null;
  return <div aria-label={`${ph.label} 비교`}>
    <p className="venue-muted"><b>{ph.label}</b> · {ph.purpose}</p>
    {ph.cameras.map((pc) => {
      const dc = during(pc.compare_to);
      const pr = averageRates(pc.gate), dr = averageRates(dc?.gate ?? null);
      const isX = pc.axis === 'x';
      const rows: [string, number | undefined, number | undefined][] = isX
        ? [['왼쪽으로 지나감', pr?.minus, dr?.minus], ['오른쪽으로 지나감', pr?.plus, dr?.plus], ['합계', pr && pr.minus + pr.plus, dr ? dr.minus + dr.plus : undefined]]
        : [['들어감(카메라에서 멀어짐)', pr?.minus, dr?.minus], ['나감(카메라 쪽으로)', pr?.plus, dr?.plus], ['순유출(나감 − 들어감)', pr && pr.plus - pr.minus, dr ? dr.plus - dr.minus : undefined]];
      const phaseCol = `${ph.label} (명/분)`, duringCol = '축제 중 (명/분)';
      const cmp = phase === 'pre' ? `축제 중 ÷ 축제 전` : `축제 후 ÷ 축제 중`;
      return <div key={pc.id} className="venue-settings" style={{marginTop: 8}}>
        <h4>{pc.name} ↔ {dc?.name ?? '축제 중 카메라 미연결'}</h4>
        {!pc.gate ? <p className="venue-muted">이 영상의 경계선 통과 수가 없습니다 (python -m app.gate_flow).</p> :
          <table className="w-full text-left text-xs"><thead><tr><th>구분</th><th>{phase === 'pre' ? phaseCol : duringCol}</th><th>{phase === 'pre' ? duringCol : phaseCol}</th><th>{cmp}</th></tr></thead>
            <tbody>{rows.map(([label, p, d]) => <tr key={label}><td>{label}</td><td>{f1(phase === 'pre' ? p : d)}</td><td>{f1(phase === 'pre' ? d : p)}</td>
              <td>{phase === 'pre' ? ratio(d, p) : ratio(p, d)}</td></tr>)}</tbody></table>}
        {phase === 'pre' && !dc?.gate && <p className="venue-muted">축제 중 같은 위치 카메라 결과가 아직 연결되지 않아 비교값이 비어 있습니다(분석이 끝나면 자동 연결).</p>}
        {phase === 'post' && pc.exit_zone && <PostExit time={time} cfg={cfg} timeline={timeline} initials={initials} pcGate={pc.gate} duringGate={dc?.gate ?? null} zone={pc.exit_zone} zoneName={zoneName} />}
      </div>;
    })}
  </div>;
}

function PostExit({time, cfg, timeline, initials, pcGate, duringGate, zone, zoneName}: {
  time: number; cfg: FestivalFlowConfig; timeline: FlowTimeline; initials: Record<string, number>; pcGate: FestivalFlowConfig['cameras'][number]['gate'];
  duringGate: FestivalFlowConfig['cameras'][number]['gate']; zone: string; zoneName: (id: string) => string;
}) {
  const post = exitShares(pcGate), dur = exitShares(duringGate), r = averageRates(pcGate);
  const labels = cfg.exit_labels ?? {left: '화면 왼쪽', center: '정면', right: '화면 오른쪽'};
  // 축제가 끝날 때 남은 인원 = 축제 중 타임라인 끝의 그 구역 인원(시작 인원 + 누적 순유입)
  const remaining = zoneFlowAt(timeline, zone, initials[zone] ?? 0, timeline.length - 1, 0).shown;
  const netOut = r ? r.plus - r.minus : 0;
  const minutes = minutesToEmpty(remaining, netOut);
  const exitPerMin = r?.plus ?? 0;
  const now = cumulativeAt(pcGate, time), all = cumulativeAt(pcGate, Infinity);
  const leftNow = now ? Math.max(0, remaining - (now.exits - now.entries)) : null;
  return <div style={{marginTop: 8}}>
    <h4>축제 후 얼마나 빠져나갔나</h4>
    <table className="w-full text-left text-xs"><thead><tr><th>구분</th><th>지금까지 (축제 후 {Math.floor(time / 60)}분 {Math.floor(time % 60)}초)</th><th>영상 전체 ({Math.round((pcGate?.seconds ?? 0) / 60)}분)</th></tr></thead>
      <tbody>
        <tr><td><b>빠져나감</b></td><td><b>{now?.exits ?? 0}명</b></td><td>{all?.exits ?? 0}명</td></tr>
        <tr><td>들어옴</td><td>{now?.entries ?? 0}명</td><td>{all?.entries ?? 0}명</td></tr>
        <tr><td>순유출 (빠져나감 − 들어옴)</td><td>{now ? now.exits - now.entries : 0}명</td><td>{all ? all.exits - all.entries : 0}명</td></tr>
        {EXIT_SIDES.map((k) => <tr key={k}><td>└ {labels[k]}으로 빠져나감</td><td>{now?.sides?.[k] ?? 0}명</td><td>{all?.sides?.[k] ?? 0}명</td></tr>)}
        <tr><td>{zoneName(zone)} 남은 인원 (추정)</td><td>{leftNow == null ? '—' : `약 ${Math.round(leftNow)}명`}</td><td>축제 종료 시 약 {Math.round(remaining)}명</td></tr>
      </tbody></table>
    <p className="venue-muted">방향별 합은 &lsquo;빠져나감&rsquo;보다 적을 수 있습니다(1초 이상 추적되며 화면 아래로 충분히 이동한 사람만 방향을 판정). 남은 인원은 축제 중 흐름 계산의 끝값(시작 인원 가정 포함)에서 순유출을 뺀 값입니다.</p>
    <h4>어느 쪽으로 빠져나가나 (방향을 판정한 {post?.total ?? 0}명 기준)</h4>
    {!post ? <p className="venue-muted">방향 데이터가 없습니다 (python -m app.gate_flow ... --exit-sides).</p> :
      <table className="w-full text-left text-xs"><thead><tr><th>방향</th><th>축제 후</th><th>축제 중</th><th>10초 / 30초 / 60초 뒤 예상 유출</th></tr></thead>
        <tbody>{EXIT_SIDES.map((k) => <tr key={k}><td>{labels[k]}</td>
          <td><span style={{display: 'inline-block', width: `${Math.round(post.shares[k] * 80)}px`, height: 8, background: '#f59e0b', marginRight: 6}} />{Math.round(post.shares[k] * 100)}% ({post.counts[k]}명)</td>
          <td>{dur ? `${Math.round(dur.shares[k] * 100)}%` : '—'}</td>
          <td>{[10, 30, 60].map((h) => (post.shares[k] * exitPerMin / 60 * h).toFixed(1)).join(' / ')}명</td></tr>)}</tbody></table>}
    <ul className="venue-actions">
      <li>축제 후 나가는 속도 분당 {f1(r?.plus)}명, 들어오는 속도 분당 {f1(r?.minus)}명 → 순유출 분당 {f1(netOut)}명</li>
      <li>축제가 끝날 때 {zoneName(zone)}에 남은 인원(축제 중 흐름 계산 끝값) 약 {Math.round(remaining)}명 → {minutes == null ? '순유출이 없어 줄지 않음' : `지금 순유출 속도면 약 ${Math.round(minutes)}분 뒤 비워짐`}</li>
    </ul>
    <p className="venue-muted">{cfg.exit_labels_note} 예상 유출은 축제 후 영상의 평균 속도와 방향 비율이 유지된다는 가정입니다. 남은 인원은 시작 인원 가정에 따라 달라집니다.</p>
  </div>;
}
