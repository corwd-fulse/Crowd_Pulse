import {memo,useEffect,useLayoutEffect,useMemo,useRef,useState} from 'react';
import type {RefObject} from 'react';
import {Canvas,useFrame,useThree} from '@react-three/fiber';
import {OrbitControls,Line} from '@react-three/drei';
import * as THREE from 'three';
import type {RoiSnapshot} from '../../services/api';
import {imageToFloor} from '../../services/replayScene';
import type {ReplayScene,ReplayActor} from '../../services/replayScene';

const palette=['#23cfff','#ffc653','#23e3bb','#ff8db3','#ab9bff'];
type CameraAction={kind:'reset'|'left'|'right'|'in'|'out';id:number};
type Controls={target:THREE.Vector3;update:()=>void};
// Three owns these mutable camera objects; update its projection imperatively.
function setCameraZoom(camera:THREE.Camera,zoom:number){
  if(camera instanceof THREE.OrthographicCamera){camera.zoom=zoom;camera.updateProjectionMatrix();}
}

function CameraRig({action}:{action:CameraAction}) {
  const {camera,size,invalidate}=useThree();
  const controls=useThree(s=>s.controls) as unknown as Controls|undefined;
  useLayoutEffect(()=>{
    setCameraZoom(camera,Math.min(size.width/26,size.height/17));invalidate();
  },[camera,size.width,size.height,invalidate]);
  useLayoutEffect(()=>{
    if(!controls)return;
    if(action.kind==='reset'){
      controls.target.set(0,0,1.6);camera.position.set(16,19,25);
      setCameraZoom(camera,Math.min(size.width/26,size.height/17));
    }else if(action.kind==='left'||action.kind==='right'){
      const offset=camera.position.clone().sub(controls.target).applyAxisAngle(new THREE.Vector3(0,1,0),(action.kind==='left'?1:-1)*Math.PI/12);
      camera.position.copy(controls.target).add(offset);
    }else if(camera instanceof THREE.OrthographicCamera){setCameraZoom(camera,THREE.MathUtils.clamp(camera.zoom*(action.kind==='in'?1.25:.8),5,100));}
    camera.lookAt(controls.target);camera.updateProjectionMatrix();controls.update();invalidate();
    // Resize has its own fit effect and must not reapply the last rotation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[action,controls,camera,invalidate]);
  return null;
}

const ZonePlatform=memo(function ZonePlatform({zone,resolution,index}:{zone:RoiSnapshot['zones'][number];resolution:[number,number];index:number}) {
  const color=zone.color??palette[index%palette.length];
  const floor=useMemo(()=>zone.polygon.map(([x,y])=>imageToFloor(x,y,resolution)),[zone.polygon,resolution]);
  const geometry=useMemo(()=>{
    const shape=new THREE.Shape(floor.map(([x,,z])=>new THREE.Vector2(x,-z)));
    return new THREE.ExtrudeGeometry(shape,{depth:.035,bevelEnabled:false});
  },[floor]);
  useLayoutEffect(()=>()=>geometry.dispose(),[geometry]);
  // Native scene labels avoid a nested React DOM root when switching 2D/3D.
  const label=useMemo(()=>{
    const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d')!;
    canvas.width=64;canvas.height=64;
    ctx.fillStyle='#101e30';ctx.beginPath();ctx.roundRect(2,2,60,60,18);ctx.fill();
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
    ctx.font='bold 32px sans-serif';ctx.fillStyle=color;ctx.textBaseline='middle';ctx.textAlign='center';ctx.fillText(index<26?String.fromCharCode(65+index):String(index+1),32,33);
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
    return {texture,width:.72};
  },[index,color]);
  useLayoutEffect(()=>()=>label.texture.dispose(),[label]);
  return <group>
    <mesh geometry={geometry} rotation={[-Math.PI/2,0,0]} position={[0,.025,0]}><meshStandardMaterial color={color} transparent opacity={.20} roughness={.9}/></mesh>
    <Line points={[...floor,floor[0]].map(([x,,z])=>[x,.09,z] as [number,number,number])} color={color} lineWidth={2}/>
    <sprite position={[floor[0][0],.65,floor[0][2]]} scale={[label.width,.72,1]}><spriteMaterial map={label.texture} transparent depthTest={false}/></sprite>
  </group>;
},(a,b)=>a.zone===b.zone&&a.index===b.index&&a.resolution[0]===b.resolution[0]&&a.resolution[1]===b.resolution[1]);

function People({actors,time,host,roi,onHover,capacity}:{actors:ReplayActor[];time:number;host:RefObject<HTMLDivElement|null>;roi:RoiSnapshot;onHover:(actor:ReplayActor|null)=>void;capacity:number}) {
  const anchors=useRef<THREE.InstancedMesh>(null);
  const meshes=useRef<(THREE.InstancedMesh|null)[]>([]);
  const invalidate=useThree(s=>s.invalidate);
  const geometries=useMemo(()=>[
    new THREE.SphereGeometry(.135,8,6),new THREE.CylinderGeometry(.18,.14,.43,8),
    new THREE.BoxGeometry(.105,.42,.115),new THREE.BoxGeometry(.105,.42,.115),
    new THREE.BoxGeometry(.10,.39,.13),new THREE.BoxGeometry(.10,.39,.13),
  ],[]);
  useLayoutEffect(()=>()=>geometries.forEach(g=>g.dispose()),[geometries]);
  useLayoutEffect(()=>{
    const matrix=new THREE.Matrix4(),base=new THREE.Matrix4(),local=new THREE.Matrix4(),rotation=new THREE.Matrix4();
    const color=new THREE.Color();
    for(let i=0;i<actors.length;i++){
      const a=actors[i];const zoneIndex=roi.zones.findIndex(z=>z.name===a.zone);
      const shirt=a.uncertain?'#efac7a':roi.zones[zoneIndex]?.color??['#7dd3fc','#c4b5fd','#5eead4','#e2e8f0'][a.trackId%4];
      base.makeRotationY(a.heading).scale(new THREE.Vector3(1.3,1.3,1.3));base.setPosition(a.floor[0],.10,a.floor[2]);
      const phase=a.moving?Math.sin(time*8+a.trackId)*.22:0;
      const offsets=[[0,1.02,0],[0,.66,0],[-.24,.64,0],[.24,.64,0],[-.09,.23,0],[.09,.23,0]];
      geometries.forEach((_,part)=>{
        local.makeTranslation(...offsets[part] as [number,number,number]);
        rotation.makeRotationX(part>=2?(part%2?-phase:phase):0);
        matrix.copy(base).multiply(local).multiply(rotation);
        meshes.current[part]?.setMatrixAt(i,matrix);
        color.set(part===0?'#f1d4bd':part>=4?'#24364b':shirt);
        meshes.current[part]?.setColorAt(i,color);
      });
      matrix.makeRotationX(-Math.PI/2);matrix.setPosition(a.floor[0],.102,a.floor[2]);anchors.current?.setMatrixAt(i,matrix);
      color.set(shirt);anchors.current?.setColorAt(i,color);
    }
    for(const mesh of [...meshes.current,anchors.current])if(mesh){mesh.count=actors.length;mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;}
    // Read actual Three instance matrices for browser parity checks, not just props.
    const rendered=actors.map((a,i)=>{anchors.current?.getMatrixAt(i,matrix);return {id:a.trackId,x:matrix.elements[12],z:matrix.elements[14]};});
    host.current?.setAttribute('data-rendered-actors',JSON.stringify(rendered));
    invalidate();
  },[actors,time,roi,geometries,host,invalidate]);
  return <group>
    <instancedMesh key={'anchors'+capacity} ref={anchors} args={[undefined,undefined,capacity]} frustumCulled={false}>
      <circleGeometry args={[.21,12]}/><meshBasicMaterial transparent opacity={.55}/>
    </instancedMesh>
    {geometries.map((geometry,i)=><instancedMesh key={`${i}-${capacity}`} ref={mesh=>{meshes.current[i]=mesh;}} args={[geometry,undefined,capacity]} frustumCulled={false}
      onPointerOver={event=>{event.stopPropagation();onHover(actors[event.instanceId??-1]??null);}}
      onPointerOut={()=>onHover(null)}>
      <meshStandardMaterial flatShading roughness={.85}/>
    </instancedMesh>)}
  </group>;
}

const FlowArrow=memo(function FlowArrow({from,to,resolution,reverse}:{from:[number,number];to:[number,number];resolution:[number,number];reverse:boolean}) {
  const a=imageToFloor(...from,resolution),b=imageToFloor(...to,resolution);
  const dx=b[0]-a[0],dz=b[2]-a[2],length=Math.hypot(dx,dz),color=reverse?'#ff8db3':'#ffc653';
  if(length<1e-6)return null;
  return <group>
    <Line points={[[a[0],.17,a[2]],[b[0],.17,b[2]]]} color={color} lineWidth={2.5}/>
    <mesh position={[b[0],.17,b[2]]} rotation={[Math.PI/2,0,-Math.atan2(dx,dz)]}><coneGeometry args={[.12,.32,6]}/><meshBasicMaterial color={color}/></mesh>
  </group>;
});

function RenderProbe({host}:{host:RefObject<HTMLDivElement|null>}) {
  const {gl,camera}=useThree();
  useFrame(()=>{
    const started=performance.now();
    queueMicrotask(()=>{
      const node=host.current;if(!node)return;
      node.dataset.renderCount=String(Number(node.dataset.renderCount??0)+1);
      node.dataset.drawCalls=String(gl.info.render.calls);node.dataset.triangles=String(gl.info.render.triangles);
      node.dataset.renderSubmitMs=String(performance.now()-started);
      node.dataset.camera=JSON.stringify({position:camera.position.toArray(),zoom:camera.zoom});
    });
  });
  return null;
}

type WorldProps={scene:ReplayScene;roi:RoiSnapshot;resolution:[number,number];time:number;capacity:number;host:RefObject<HTMLDivElement|null>;onHover:(actor:ReplayActor|null)=>void;action:CameraAction;active:boolean};
const ReplayWorld=memo(function ReplayWorld({scene,roi,resolution,time,capacity,host,onHover,action}:WorldProps){
  const depth=20*resolution[1]/resolution[0];
  return <>
    <color attach="background" args={['#0b1422']}/>
    <ambientLight intensity={1.6}/><directionalLight position={[-3,14,6]} intensity={2}/><hemisphereLight args={['#e0f2fe','#182337',1.2]}/>
    <mesh position={[0,-.34,0]}><boxGeometry args={[20.35,.12,depth+.35]}/><meshStandardMaterial color="#101c2c" roughness={1}/></mesh>
    <mesh position={[0,-.13,0]}><boxGeometry args={[20,.28,depth]}/><meshStandardMaterial color="#24364a" roughness={1}/></mesh>
    <gridHelper args={[20,10,'#405166','#35465b']} scale={[1,1,depth/20]} position={[0,.015,0]}/>
    {roi.zones.map((z,i)=>z.visible===false?null:<ZonePlatform key={z.name} zone={z} index={i} resolution={resolution}/>)}
    <People actors={scene.actors} roi={roi} time={time} host={host} onHover={onHover} capacity={capacity}/>
    {scene.events.map(e=><FlowArrow key={e.event_id} from={e.from_position} to={e.to_position} resolution={resolution} reverse={e.from_zone.localeCompare(e.to_zone)>0}/>)}
    <OrbitControls makeDefault enableDamping={false} minZoom={5} maxZoom={100} minPolarAngle={.2} maxPolarAngle={1.45}/>
    <CameraRig action={action}/><RenderProbe host={host}/>
  </>;
},(a,b)=>!b.active||(Object.keys(a) as (keyof WorldProps)[]).every(key=>Object.is(a[key],b[key])));

export default function Replay3D({scene,roi,resolution,time,onFallback,capacity,fitViewport=false}:{scene:ReplayScene;roi:RoiSnapshot;resolution:[number,number];time:number;onFallback:(reason:string)=>void;capacity:number;fitViewport?:boolean}) {
  const host=useRef<HTMLDivElement>(null);
  const [action,setAction]=useState<CameraAction>({kind:'reset',id:0});
  const [hover,setHover]=useState<ReplayActor|null>(null),[low,setLow]=useState(true);
  const [active,setActive]=useState(true);
  useEffect(()=>{
    const node=host.current;if(!node)return;
    let intersecting=true;
    const update=()=>setActive(intersecting&&!document.hidden);
    const observer=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(entries=>{intersecting=entries[0]?.isIntersecting??false;update();});
    observer?.observe(node);document.addEventListener('visibilitychange',update);update();
    return()=>{observer?.disconnect();document.removeEventListener('visibilitychange',update);};
  },[]);
  return <div aria-label="3D 아이소메트릭 관측 시뮬레이션" ref={host} data-three-active={active} data-three-frame={scene.frame??''} data-three-actors={scene.actors.length} data-three-events={JSON.stringify(scene.events.map(e=>e.event_id))} className={`relative overflow-hidden bg-[#0b1422] ${fitViewport?'h-full':'rounded-xl border border-slate-600/40'}`}>
    <div className={`flex flex-wrap items-center justify-between gap-3 px-4 py-3 ${fitViewport?'pointer-events-none absolute inset-x-0 top-0 z-10':'border-b border-white/5'}`}>
      <div><span className="text-[10px] font-semibold tracking-[.2em] text-cyan-300/70">SPATIAL REPLAY</span><h4 className="mt-1 text-sm font-semibold text-slate-100">인파 디지털 트윈 <span className="ml-2 text-[10px] font-normal text-slate-500">관측 기록</span></h4></div>
      <div className="text-right"><span className="text-2xl font-semibold tabular-nums text-slate-100">{scene.actors.length}<span className="ml-1 text-xs font-normal text-slate-400">명</span></span><p className="mt-1 text-[10px] text-slate-500">프레임 {scene.frame??'없음'}</p></div>
    </div>
    <div className={`pointer-events-none flex flex-wrap gap-2 px-4 pt-3 ${fitViewport?'absolute left-0 top-16 z-10':''}`}>{roi.zones.map((zone,index)=>zone.visible===false?null:<span key={zone.name} className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[.03] px-2 py-1 text-[10px] text-slate-300"><span className="font-bold" style={{color:zone.color??palette[index%palette.length]}}>{index<26?String.fromCharCode(65+index):index+1}</span>{zone.name}<span className="text-slate-500">{scene.actors.filter(a=>a.zone===zone.name).length}명</span></span>)}</div>
    <div className={`relative min-w-0 ${fitViewport?'h-full':'h-[360px] sm:h-[430px]'}`} style={{touchAction:'none'}}>
      <Canvas orthographic camera={{position:[18,18,18],zoom:15,near:.1,far:200}} frameloop={active?'demand':'never'} dpr={low?1:[1,1.5]}
        gl={{antialias:false,alpha:false,powerPreference:'low-power'}}
        onCreated={({gl})=>{
          gl.domElement.setAttribute('aria-label','실제 Three.js WebGL 3D 화면');
          gl.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();onFallback('3D 그래픽 연결이 끊겨 같은 시각의 2D로 전환했습니다.');},{once:true});
          const context=gl.getContext(),extension=context.getExtension('WEBGL_debug_renderer_info');
          if(host.current)host.current.dataset.renderer=extension?String(context.getParameter(extension.UNMASKED_RENDERER_WEBGL)):'WebGL2';
        }}>
        <ReplayWorld scene={scene} roi={roi} resolution={resolution} time={time} host={host} onHover={setHover} capacity={capacity} action={action} active={active}/>
      </Canvas>
      {hover&&scene.actors.some(a=>a.trackId===hover.trackId)&&<div className="pointer-events-none absolute left-2 top-2 rounded-sm bg-[#061526]/90 p-2 text-xs">ID {hover.trackId} · {hover.zone||'구역 밖'} · 원본 ({hover.x.toFixed(1)}, {hover.y.toFixed(1)})</div>}
      {scene.frame===null&&<p className="absolute bottom-3 left-3 text-xs text-warn">관측 프레임 없음 — 캐릭터 미표시</p>}
    </div>
    <div className={`flex flex-wrap items-center gap-2 border-t border-white/5 px-3 py-2 text-[10px] text-slate-300 ${fitViewport?'absolute inset-x-0 bottom-0 z-10 bg-[#0b1422]/90':''}`}>
      {([['left','3D 좌회전'],['right','3D 우회전'],['in','3D 확대'],['out','3D 축소'],['reset','3D 기본 시점']] as const).map(([kind,label])=><button key={kind} type="button" className="rounded-md border border-white/10 bg-white/[.03] px-2.5 py-2 transition-colors hover:border-cyan-300/40 hover:bg-cyan-300/10" onClick={()=>setAction(a=>({kind,id:a.id+1}))}>{label}</button>)}
      <label className="ml-auto flex items-center gap-1.5 text-slate-400"><input type="checkbox" checked={low} onChange={e=>setLow(e.target.checked)}/> 저사양 모드</label>
    </div>
    <p className={`${fitViewport?'sr-only':'px-4 pb-3'} text-[10px] leading-relaxed text-slate-500`}>드래그로 회전 · 휠로 확대 · 살구색 캐릭터: 저신뢰 관측<br/>영상 좌표 기반 개념적 공간 · 실제 거리와 캐릭터 크기를 나타내지 않습니다.</p>
  </div>;
}
