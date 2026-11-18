import type {VenueConfiguration} from '../../services/venueControl';
import {center,value} from '../../services/venueControl';
import type {ConcentrationResult} from '../../services/concentration';

export default function ConcentrationMapLayer({config,result}:{config:VenueConfiguration;result:ConcentrationResult|null}){
  const central=config.zones.find(z=>z.id===config.center_zone_id),ratio=config.width/config.height;
  return <g aria-label="다중 카메라 관측 오버레이" style={{pointerEvents:'none'}}>
    <g aria-label="등록된 구역 연결 관계">{config.links?.map(link=>{const a=config.zones.find(z=>z.id===link.from_zone_id),b=config.zones.find(z=>z.id===link.to_zone_id);if(!a?.visible||!b?.visible)return null;const from=center(a.polygon),to=center(b.polygon);return <g key={link.id}><line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} stroke="#738398" strokeWidth="1" strokeDasharray="4 8" vectorEffect="non-scaling-stroke"/><text transform={`translate(${(from[0]+to[0])/2} ${(from[1]+to[1])/2}) scale(1 ${ratio})`} textAnchor="middle" fill="#9aaabd" fontSize="14">{link.name} (설정)</text></g>;})}</g>
    <defs><marker id="central-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#edbd6e"/></marker></defs>
    {central?.visible&&<><polygon points={central.polygon.map(p=>p.join(',')).join(' ')} fill="none" stroke={result?.state==='긴급 확인 필요'?'#ec8176':'#e0ba78'} strokeWidth="3" strokeDasharray="12 6" vectorEffect="non-scaling-stroke"/>{(()=>{const p=center(central.polygon);return <text transform={`translate(${p[0]} ${p[1]+55}) scale(1 ${ratio})`} textAnchor="middle" fill="#f0cf91" fontSize="20">중앙 집결 · {result?value(result.observed_center_count):'미관측'}</text>;})()}</>}
    {(config.cameras??[]).filter(c=>c.position_confirmed).map(c=>{
      const count=result?.cameras.find(r=>r.camera_id===c.camera_id)?.count;
      return <g key={c.camera_id} opacity={c.active===false?0.35:1}>{c.coverage.length>=3&&<polygon points={c.coverage.map(p=>p.join(',')).join(' ')} stroke="#88b9d0" fill="#88b9d0" fillOpacity=".06" strokeDasharray="5 5" strokeWidth="1" vectorEffect="non-scaling-stroke"/>}<circle cx={c.point[0]} cy={c.point[1]} r="12" fill="#26495d" stroke="#b0d4e5" strokeWidth="2" vectorEffect="non-scaling-stroke"/>{c.bearing_deg!==null&&<line x1={c.point[0]} y1={c.point[1]} x2={c.point[0]+Math.sin(c.bearing_deg*Math.PI/180)*35} y2={c.point[1]-Math.cos(c.bearing_deg*Math.PI/180)*35} stroke="#b0d4e5" strokeWidth="2" vectorEffect="non-scaling-stroke"/>}<text transform={`translate(${c.point[0]>=700?c.point[0]-18:c.point[0]+18} ${Math.min(980,c.point[1]+75)}) scale(1 ${ratio})`} textAnchor={c.point[0]>=700?'end':'start'} fill="#c5e3f0" fontSize="18">{c.name}{c.active===false?' (비활성)':''} · {value(count)}</text></g>;
    })}
    {result?.routes.flatMap(r=>{
      const origin=config.zones.find(z=>z.id===r.from_zone_id),destination=config.zones.find(z=>z.id===r.to_zone_id);
      if(!origin||!destination||!origin.visible||!destination.visible)return [];
      return (['toward_center','away_from_center'] as const).flatMap((direction,i)=>{
        const n=r[direction];if(n===null||n<=0)return [];
        const a=center(i?destination.polygon:origin.polygon),b=center(i?origin.polygon:destination.polygon);
        const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy)||1,ox=-dy/len*20,oy=dx/len*20;
        const x=a[0]+dx*.2+ox,y=a[1]+dy*.2+oy,tx=b[0]-dx*.2+ox,ty=b[1]-dy*.2+oy;
        return [<g key={`${r.corridor_id}-${direction}`} data-central-corridor={r.corridor_id} data-central-direction={direction}><path d={`M${x},${y} L${tx},${ty}`} fill="none" stroke="#edbd6e" strokeDasharray={r.quality==='unavailable'?'5 5':undefined} strokeWidth={Math.min(9,1.5+Math.sqrt(n))} vectorEffect="non-scaling-stroke" markerEnd="url(#central-arrow)"/><text transform={`translate(${(x+tx)/2} ${(y+ty)/2-15}) scale(1 ${ratio})`} textAnchor="middle" fill="#f0d398" fontSize="17">{r.name} · {i?'반대':'중앙 방향'} {n}회</text></g>];
      });
    })}
  </g>;
}
