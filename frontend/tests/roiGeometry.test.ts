import {describe,expect,it} from 'vitest';
import {screenToVideo,validateZones} from '../src/services/roiGeometry';
import type {Point} from '../src/services/roiGeometry';
const square:Point[]=[[10,10],[100,10],[100,100],[10,100]];
const check=(p:Point[])=>validateZones([{name:'구역',polygon:p}],[1920,1080]);
describe('original video coordinates',()=>{
  it.each([[1920,1080,960,540],[1280,720,640,360],[1080,1920,540,960]])('maps letterbox-free wrapper center at %i × %i',(w,h,x,y)=>{
    expect(screenToVideo(205,160,{left:10,top:20,width:390,height:280},[w,h])).toEqual([x,y]);
  });
  it('rejects outside and zero-size surfaces and preserves exact edges',()=>{
    const r={left:20,top:30,width:960,height:540};
    expect(screenToVideo(980,570,r,[1920,1080])).toEqual([1920,1080]);
    expect(screenToVideo(19,30,r,[1920,1080])).toBeNull();
    expect(screenToVideo(20,30,{...r,width:0},[1920,1080])).toBeNull();
  });
});
describe('polygon validation',()=>{
  it('accepts a simple concave polygon',()=>expect(check([[10,10],[100,10],[50,50],[100,100],[10,100]])).toEqual([]));
  it('rejects a bow tie',()=>expect(check([[0,0],[100,100],[0,100],[100,0]]).join()).toContain('자기 교차'));
  it('rejects zero area and repeated vertices',()=>{
    expect(check([[0,0],[1,1],[2,2]]).length).toBeGreaterThan(0);
    expect(check([[0,0],[100,0],[100,0],[0,100]]).join()).toContain('중복');
  });
  it('rejects backtracking and nonfinite/outside coordinates',()=>{
    expect(check([[0,0],[100,0],[50,0],[100,100],[0,100]]).join()).toContain('되짚어');
    expect(check([[0,0],[2000,0],[0,100]]).join()).toContain('범위');
    expect(check([[0,0],[NaN,0],[0,100]]).join()).toContain('범위');
  });
  it.each([
    [[50,50],[150,50],[150,150],[50,150]],
    [[20,20],[30,20],[30,30],[20,30]],
    [[100,10],[200,10],[200,100],[100,100]],
  ] as Point[][])('rejects intersection, containment, and touching boundaries',(...p)=>{
    expect(validateZones([{name:'A',polygon:square},{name:'B',polygon:p}],[1920,1080]).join()).toContain('겹치');
  });
  it('accepts separated regions and rejects duplicate trimmed names',()=>{
    const zones=[{name:'A',polygon:square},{name:'B',polygon:square.map(([x,y]):Point=>[x+200,y])}];
    expect(validateZones(zones,[1920,1080])).toEqual([]);
    expect(validateZones([zones[0],{...zones[1],name:' A '}],[1920,1080]).join()).toContain('이름');
  });
});
