import {Component} from 'react';
import type {ReactNode} from 'react';

export default class Replay3DBoundary extends Component<{children:ReactNode;onFallback:(reason:string)=>void},{failed:boolean}> {
  state={failed:false};
  static getDerivedStateFromError(){return {failed:true};}
  componentDidCatch(){this.props.onFallback('3D 렌더러를 사용할 수 없어 같은 시각의 2D로 전환했습니다.');}
  render(){return this.state.failed?<p role="status">2D 리플레이로 복귀하는 중…</p>:this.props.children;}
}
