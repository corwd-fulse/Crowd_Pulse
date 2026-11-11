import { useState } from 'react';

const zones = [
  { id: 'A', name: '중앙 출입구', count: 125, congestion: 88, color: '#ff527f', x: 125, y: 145 },
  { id: 'B', name: '메인 통로', count: 78, congestion: 65, color: '#ffc653', x: 380, y: 100 },
  { id: 'C', name: '야외 광장', count: 45, congestion: 32, color: '#23e3bb', x: 620, y: 210 },
];

function project(x: number, y: number, z = 0) {
  return `${440 + (x - y) * 0.72},${135 + (x + y) * 0.28 - z}`;
}

function Building({ x, y, w, d, h, color }: { x: number; y: number; w: number; d: number; h: number; color: string }) {
  return (
    <g stroke={color} strokeWidth="0.8">
      <polygon points={`${project(x, y)} ${project(x + w, y)} ${project(x + w, y + d)} ${project(x, y + d)}`} fill={color} fillOpacity="0.08" />
      <polygon points={`${project(x, y, h)} ${project(x + w, y, h)} ${project(x + w, y + d, h)} ${project(x, y + d, h)}`} fill={color} fillOpacity="0.2" />
      <polygon points={`${project(x, y)} ${project(x + w, y)} ${project(x + w, y, h)} ${project(x, y, h)}`} fill={color} fillOpacity="0.11" />
      <polygon points={`${project(x + w, y)} ${project(x + w, y + d)} ${project(x + w, y + d, h)} ${project(x + w, y, h)}`} fill={color} fillOpacity="0.23" />
      {Array.from({ length: Math.floor(h / 9) }, (_, i) => (
        <polyline key={i} points={`${project(x, y, i * 9)} ${project(x + w, y, i * 9)} ${project(x + w, y + d, i * 9)}`} fill="none" opacity="0.5" />
      ))}
      {Array.from({ length: Math.floor(w / 12) }, (_, i) => (
        <line key={i} x1={440 + (x + i * 12 - y) * 0.72} y1={135 + (x + i * 12 + y) * 0.28} x2={440 + (x + i * 12 - y) * 0.72} y2={135 + (x + i * 12 + y) * 0.28 - h} opacity="0.4" />
      ))}
    </g>
  );
}

export default function FacilityMap() {
  const [selected, setSelected] = useState('A');
  const zone = zones.find((item) => item.id === selected)!;

  return (
    <section className="facility-map" aria-labelledby="facility-title">
      <div className="technical-heading">
        <h2 id="facility-title">시설 내 인파 분포</h2>
        <span>공간 구성 예시 · 샘플 데이터</span>
      </div>
      <div className="relative">
        <div className="map-readout">
          <span className="text-[10px] tracking-[0.18em] text-muted">ZONE {zone.id} / MONITORING</span>
          <h3 className="mt-2 text-lg font-semibold">{zone.name}</h3>
          <div className="mt-4 flex gap-8">
            <div><p className="text-[11px] text-muted">탐지 인원</p><p className="num mt-1 text-3xl" style={{ color: zone.color }}>{zone.count}<small className="ml-1 text-xs">명</small></p></div>
            <div><p className="text-[11px] text-muted">혼잡 지표</p><p className="num mt-1 text-3xl" style={{ color: zone.color }}>{zone.congestion}<small className="ml-1 text-xs">%</small></p></div>
          </div>
        </div>
        <svg viewBox="0 0 900 440" className="facility-drawing" role="img" aria-label="중앙 출입구, 메인 통로, 야외 광장의 위치를 표현한 시연용 입체 시설도">
          <defs>
            <radialGradient id="map-halo"><stop stopColor="#086dcc" stopOpacity="0.28"/><stop offset="1" stopColor="#086dcc" stopOpacity="0"/></radialGradient>
            <filter id="map-glow"><feGaussianBlur stdDeviation="3"/></filter>
            <linearGradient id="platform-face" x2="0" y2="1"><stop stopColor="#17477d" stopOpacity="0.6"/><stop offset="1" stopColor="#081937" stopOpacity="0.2"/></linearGradient>
          </defs>
          <ellipse cx="460" cy="285" rx="390" ry="150" fill="url(#map-halo)"/>
          <path d="M80 284 440 144 826 294 466 434Z" fill="url(#platform-face)" stroke="#24618a" strokeWidth="0.7"/>
          <path d="M80 284v8l386 150 360-140v-8M466 434v8" fill="none" stroke="#28628d" opacity="0.65"/>
          <g stroke="#2167a1" strokeWidth="0.5" opacity="0.4">
            {Array.from({ length: 25 }, (_, i) => <g key={i}><line x1={440 + (i * 25) * .72} y1={135 + i * 25 * .28} x2={440 + (i * 25 - 600) * .72} y2={135 + (i * 25 + 600) * .28}/><line x1={440 - i * 25 * .72} y1={135 + i * 25 * .28} x2={440 + (600 - i * 25) * .72} y2={135 + (600 + i * 25) * .28}/></g>)}
          </g>
          <g fill="none" strokeWidth="3">
            <path d="M180 290 300 245 450 305 620 240 745 290" stroke="#16c9e8" opacity="0.65"/>
            <path d="M300 245 390 210 480 245 590 200" stroke="#f5b844" opacity="0.75"/>
            <path d="M450 305 380 335 260 290" stroke="#eb4580" opacity="0.7"/>
          </g>
          <g fill="none" stroke={zone.color} className="selected-zone-ring">
            <ellipse cx={selected === 'A' ? 218 : selected === 'B' ? 540 : 716} cy={selected === 'A' ? 280 : selected === 'B' ? 212 : 280} rx="53" ry="19" strokeWidth="7" opacity="0.16" filter="url(#map-glow)"/>
            <ellipse cx={selected === 'A' ? 218 : selected === 'B' ? 540 : 716} cy={selected === 'A' ? 280 : selected === 'B' ? 212 : 280} rx="53" ry="19" strokeWidth="1"/>
            <ellipse cx={selected === 'A' ? 218 : selected === 'B' ? 540 : 716} cy={selected === 'A' ? 280 : selected === 'B' ? 212 : 280} rx="60" ry="22" strokeDasharray="4 5" opacity="0.6"/>
          </g>
          <Building x={150} y={200} w={165} d={120} h={85} color="#1fb9ff"/>
          <Building x={170} y={225} w={45} d={65} h={125} color="#29d7f5"/>
          <Building x={235} y={230} w={45} d={65} h={105} color="#29d7f5"/>
          <Building x={30} y={330} w={100} d={55} h={35} color="#ff4583"/>
          <Building x={150} y={35} w={130} d={60} h={45} color="#f9bd44"/>
          <Building x={380} y={100} w={125} d={100} h={55} color="#21d6a2"/>
          <Building x={430} y={340} w={110} d={75} h={35} color="#5b83ff"/>
          {Array.from({ length: 32 }, (_, i) => <g key={i} fill={i < 16 ? '#ff6b98' : '#32ead0'} opacity="0.8"><circle cx={i < 16 ? 200 + (i % 8) * 8 : 620 + (i % 8) * 8} cy={i < 16 ? 278 + Math.floor(i / 8) * 7 : 270 + Math.floor((i - 16) / 8) * 7} r="1.7"/></g>)}
          <g fill="#c1e8ff" fontSize="12" textAnchor="middle">
            <text x="218" y="246">A · 중앙 출입구</text>
            <text x="540" y="150">B · 메인 통로</text>
            <text x="716" y="248">C · 야외 광장</text>
            <text x="390" y="182" fill="#4adfff">중앙 관제 시설</text>
            <text x="555" y="368" fill="#739cdd">보조 출입구</text>
          </g>
          <g stroke="#3e769b" fill="none" strokeWidth="0.7">
            <path d="M218 251v20m322-115v43m176 54v17M391 187l-19 13"/>
          </g>
        </svg>
        <div className="map-caption"><span>시설 배치도 · 개념 시각화</span><span>구역을 선택해 인파 현황 확인</span></div>
        <div className="map-zone-controls" aria-label="시설 구역 선택">
          {zones.map((item) => <button key={item.id} type="button" aria-pressed={selected === item.id} onClick={() => setSelected(item.id)} style={{ '--zone-color': item.color } as React.CSSProperties}><span/>{item.id}구역 <span className="hidden sm:inline">{item.name}</span><b className="num">{item.congestion}%</b></button>)}
        </div>
      </div>
    </section>
  );
}
