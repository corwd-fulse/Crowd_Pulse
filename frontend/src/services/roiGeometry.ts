import type {RoiSnapshot} from './api';
export type Point = [number, number];
export type Zone = RoiSnapshot['zones'][number];
const eps = 1e-7;
const cross = (a: Point, b: Point, c: Point) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const on = (a: Point,b: Point,p: Point) => Math.abs(cross(a,b,p))<eps && p[0]>=Math.min(a[0],b[0])-eps && p[0]<=Math.max(a[0],b[0])+eps && p[1]>=Math.min(a[1],b[1])-eps && p[1]<=Math.max(a[1],b[1])+eps;
function intersects(a: Point,b: Point,c: Point,d: Point) {
  return (cross(a,b,c)*cross(a,b,d)<0 && cross(c,d,a)*cross(c,d,b)<0) || on(a,b,c)||on(a,b,d)||on(c,d,a)||on(c,d,b);
}
function inside(p: Point, poly: Point[]) {
  let result=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++) {
    const a=poly[i],b=poly[j];
    if(on(a,b,p)) return true;
    if((a[1]>p[1])!==(b[1]>p[1]) && p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]) result=!result;
  }
  return result;
}
// SVG fills a wrapper with exactly the video's intrinsic aspect ratio: no letterbox offset.
export function screenToVideo(x: number,y: number,rect: {left:number;top:number;width:number;height:number},resolution: Point): Point | null {
  if(rect.width<=0 || rect.height<=0 || x<rect.left || y<rect.top || x>rect.left+rect.width || y>rect.top+rect.height) return null;
  return [Math.round((x-rect.left)/rect.width*resolution[0]*100)/100,Math.round((y-rect.top)/rect.height*resolution[1]*100)/100];
}
export function validateZones(zones: Zone[],resolution: Point): string[] {
  const errors: string[]=[];
  if(!resolution.every(n=>Number.isInteger(n)&&n>0)) return ['영상 해상도를 먼저 확인하세요.'];
  if(!zones.length) errors.push('구역을 하나 이상 완성하세요.');
  const names=new Set<string>();
  for(const [index,z] of zones.entries()) {
    const label=`구역 ${index+1}`;
    if(!z.name.trim() || z.name.length>100 || names.has(z.name.trim())) errors.push(`${label}: 이름은 중복 없이 1~100자로 입력하세요.`);
    names.add(z.name.trim());
    const p=z.polygon;
    if(p.length<3) {errors.push(`${label}: 꼭짓점이 3개 이상 필요합니다.`);continue;}
    if(p.some(([x,y])=>!Number.isFinite(x+y)||x<0||y<0||x>resolution[0]||y>resolution[1])) errors.push(`${label}: 영상 범위를 벗어난 좌표입니다.`);
    let area=0;
    for(let i=0;i<p.length;i++) {
      const a=p[i],b=p[(i+1)%p.length];area+=a[0]*b[1]-b[0]*a[1];
      if(Math.hypot(a[0]-b[0],a[1]-b[1])<eps) errors.push(`${label}: 중복 꼭짓점입니다.`);
      // Adjacent edges may meet only at their shared endpoint (no backtracking).
      const c=p[(i+2)%p.length];
      if(Math.abs(cross(a,b,c))<eps && (on(a,b,c)||on(b,c,a))) errors.push(`${label}: 변이 되짚어 겹칩니다.`);
      for(let j=i+1;j<p.length;j++) {
        if(j===i+1 || (i===0&&j===p.length-1)) continue;
        if(intersects(a,b,p[j],p[(j+1)%p.length])) errors.push(`${label}: 자기 교차가 있습니다.`);
      }
    }
    if(Math.abs(area)<eps) errors.push(`${label}: 면적이 없는 구역입니다.`);
    for(let j=0;j<index;j++) {
      const q=zones[j].polygon;
      if(p.some((a,i)=>q.some((c,k)=>intersects(a,p[(i+1)%p.length],c,q[(k+1)%q.length]))) || (q.length>0&&(inside(p[0],q)||inside(q[0],p)))) errors.push(`${label}: 구역 ${j+1}과 겹치거나 경계가 접합니다. 간격을 두세요.`);
    }
  }
  return [...new Set(errors)];
}
