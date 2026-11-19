import {api} from './api';
import type {Point} from './roiGeometry';

export interface VenueCamera {
  camera_id:string;name:string;registered_camera_id:number|null;point:Point;position_confirmed:boolean;bearing_deg:number|null;
  active?:boolean;zone_id:string;job_id:string|null;roi_name:string;coverage:Point[];full_zone_coverage:boolean;capture_start:string|null;
  clock_verified:boolean;clock_evidence:string;scenario_offset_s:number;
}
export interface Corridor {
  corridor_id:string;name:string;camera_id:string;from_zone_id:string;to_zone_id:string;boundary:[Point,Point];center_side:Point;
  role:'approach'|'center_boundary';boundary_evidence:string;boundary_verified:boolean;counting_group:string;
  independent_counting_verified:boolean;independence_evidence:string;arrival_ratio:number|null;travel_seconds:number|null;
  arrival_basis:'unknown'|'assumption'|'measured';arrival_evidence:string;
}
export interface ConcentrationPolicy {
  window_seconds:number;center_observer_camera_id:string|null;center_boundary_coverage_complete:boolean;coverage_evidence:string;
  attention_net_per_minute:number|null;caution_capacity_ratio:number|null;urgent_capacity_ratio:number|null;
  thresholds_validated:boolean;threshold_evidence:string;
}
export const defaultPolicy:ConcentrationPolicy={window_seconds:30,center_observer_camera_id:null,center_boundary_coverage_complete:false,coverage_evidence:'',attention_net_per_minute:null,caution_capacity_ratio:null,urgent_capacity_ratio:null,thresholds_validated:false,threshold_evidence:''};
export interface CameraObservation {camera_id:string;name:string;zone_id:string;job_id:string;video_id:string|null;offset_seconds:number|null;count:number|null;count_change:number|null;reason:string|null;observed_at:string|null;analysis_mode:string;metric:{motion:number|null;flow_angle_deg:number|null;low_confidence:boolean|null;direction_counts?:Record<string,number>|null;direction_sample_count?:number|null}|null;bottleneck:{score:number;severity:string}|null}
export interface CorridorObservation {corridor_id:string;name:string;camera_id:string;from_zone_id:string;to_zone_id:string;toward_center:number|null;away_from_center:number|null;net:number|null;toward_change:number|null;window_seconds:number;quality:string;reason:string|null;arrival_basis:string;arrival_ratio:number|null;travel_seconds:number|null}
export interface ConcentrationResult {
  run_id:string;created_at:string;venue_id:string;revision:number;model_version:string;method:string;mode:string;observation_kind:string;
  observed_at:string|null;reference_offset_seconds:number;window_seconds:number;venue_name:string;registered_address:string|null;center_zone_id:string|null;
  cameras:CameraObservation[];routes:CorridorObservation[];observed_center_count:number|null;center_total_coverage_verified:boolean;
  aggregation_allowed:boolean;aggregation_reasons:string[];state:string;state_basis:string[];threshold_basis:string;increasing_corridors:number|null;
  estimates:{horizon_seconds:number;target_at:string|null;target_offset_seconds:number;expected_arrivals:number|null;expected_departures:number|null;expected_center_count:number|null;inward_pressure_per_minute:number|null;confidence:string;reasons:string[];validation_status:string}[];
  priority_corridors:{corridor_id:string;name:string;toward_center:number|null;recommendation:string}[];
}
export interface OperatorRecord {id:string;run_id:string;operator:string;zone_id:string|null;verification:string;action:string;note:string;created_at:string}
export interface Evaluation {model_version:string;mode?:string;evidence:string;five_minute_supported?:boolean;metrics:{horizon_seconds:number;samples:number;mae:number|null;rmse:number|null;status:string}[]}
export const concentrationReasons:Record<string,string>={
  independent_videos_not_aggregated:'독립 영상은 합산하지 않음',center_not_configured:'중앙 구역 미설정',no_center_corridors:'중앙 통로 미등록',
  camera_unobserved_or_unsynchronized:'카메라 미관측 또는 시각 동기화 불가',test_analysis_not_real_simultaneous_observation:'테스트 분석은 실제 동시 관측으로 통합하지 않음',
  duplicate_video_roi_source:'동일 영상·ROI 중복 연결',overlapping_counting_groups:'통로 집계 그룹 중복',counting_independence_unverified:'통로 간 중복 배제 근거 미확인',
  duplicate_crossing_evidence:'동일 원본의 경계 통과 근거 중복',
  configured_net_inflow_threshold:'등록된 순유입 검토 기준 도달',multiple_corridors_increasing:'여러 통로의 중앙 방향 통과 증가',
  stored_bottleneck_high_research_indicator:'저장된 병목 연구 지표 높음',configured_observed_capacity_ratio:'등록된 수용 대비 관측 비율 기준 도달',configured_urgent_observed_ratio:'등록된 긴급 현장 확인 비율 기준 도달',
  incomplete_gate_history:'경계 통과 이력·품질 부족',arrival_ratio_or_travel_time_unknown:'도착 비율 또는 이동 시간 미확보',center_total_unavailable:'중앙 전체 인원 추정 조건 미충족',
  insufficient_history:'관측 이력 부족',unobserved_frame:'해당 프레임 미관측',unverified_capture_clock:'촬영 시작 시각·동기화 근거 미등록',
  insufficient_arrival_history:'이동 시간과 집계 창을 포함하는 과거 관측 부족',
  tracking_or_frame_index_unavailable:'추적 또는 처리 프레임 기록 미확인',missing_frame_index:'관측 프레임 결측',unobserved_frame_or_roi:'해당 시각·ROI 미관측',
  tracking_discontinuity_or_unidentified_observation:'ID 단절·불명 관측 또는 겹침',boundary_not_verified:'경계선·중앙 방향 미확인',result_unavailable:'완료 분석 결과 없음'};
export const reasonLabel=(reason:string)=>concentrationReasons[reason]??reason;
export function estimateConcentration(revision:number,offset_seconds:number,at:string|null,signal?:AbortSignal,venueId='main'){return api.post<ConcentrationResult>(`/api/venues/${encodeURIComponent(venueId)}/concentration/estimate`,{revision,offset_seconds,at},{signal,timeout:60000}).then(r=>r.data);}
export const evaluateConcentration=(revision:number,venueId='main')=>api.post<Evaluation>(`/api/venues/${encodeURIComponent(venueId)}/concentration/evaluate`,{revision,offset_seconds:0},{timeout:120000}).then(r=>r.data);
export const getOperatorEvents=(venueId='main')=>api.get<OperatorRecord[]>(`/api/venues/${encodeURIComponent(venueId)}/operator-events`).then(r=>r.data);
export const recordOperatorEvent=(request:{run_id:string;operator:string;zone_id:string|null;verification:string;action:string;note:string},venueId='main')=>api.post(`/api/venues/${encodeURIComponent(venueId)}/operator-events`,request).then(r=>r.data);
export function incidentBrief(result:ConcentrationResult,zoneName:string,verification:string,note:string){
  return [`신고 참고 정보 · 자동 신고 아님`, `행사장: ${result.venue_name}`,`등록 주소: ${result.registered_address??'미등록 — 실제 주소 확인 필요'}`,
    `확인 구역: ${zoneName||'미선택'}`,`관측 기준: ${result.observed_at??`녹화 영상 기준 ${result.reference_offset_seconds.toFixed(2)}초 (절대 촬영 시각 미확인)`}`,
    `자료 모드: ${result.mode==='scenario'?'가상 시나리오 (실제 동시 관측 아님)':result.mode==='synchronized'?'촬영 시각 대응 등록':'독립 영상'}`,
    `관측 중앙 인원: ${result.observed_center_count??'미관측'} (전체 범위 확인: ${result.center_total_coverage_verified?'등록됨':'미확인'})`,
    `검토 상태: ${result.state} · ${result.threshold_basis==='administrator_validated'?'관리자 검증 기록 기준':'검증 전 제안 기준'}`,`근거: ${[...result.state_basis,...result.aggregation_reasons].map(reasonLabel).join(' / ')||'추가 현장 확인 필요'}`,
    `연결 카메라: ${result.cameras.map(c=>`${c.name} (${c.job_id.slice(0,8)})`).join(', ')||'미등록'}`,`현장 확인 상태: ${verification}`,`관리자 메모: ${note||'없음'}`,
    `추정 결과는 사고 확정 판정이 아닙니다. 실제 긴급 상황은 AI 결과나 화면 확인 절차를 기다리지 말고 관리자 판단으로 112 또는 119에 신고하세요.`].join('\n');
}
