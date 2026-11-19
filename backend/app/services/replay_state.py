"""Constant-memory reducer for the existing observed transition policy.

State survives every storage/page boundary. No detector or tracker is changed.
"""
import math
from .roi_geometry import signed_distance


class TrackState:
    def __init__(self, tid, polygons, emit_event, emit_diagnostic, *, min_dwell=3, hysteresis=3., displacement=15.):
        self.tid=tid;self.polygons=polygons;self.emit=emit_event;self.diagnostic=emit_diagnostic
        self.min_dwell=min_dwell;self.hysteresis=hysteresis;self.displacement=displacement
        self.segment=0;self.last=None;self.fragmented=False;self.paths=[]
        self.reset_path()

    def reset_path(self):
        self.confirmed=None;self.candidate=None;self.run=0;self.anchor=None;self.minimum=1.
        self.last_zone=None;self.steps=[];self.first_zone=None;self.path_last=None;self.path_length=0

    def reset(self):
        if self.path_length:self.paths.append((self.first_zone,self.path_last,self.path_length))
        # Only the first two segments are needed to decide whether OD is uncensored.
        self.paths=self.paths[:2]
        self.segment+=1;self.reset_path()

    def step(self,frame,t_ms,p,raw,confidence):
        if self.last is None and frame>0:self.diagnostic(dict(frame=frame,track_id=self.tid,reason='unobserved_origin'))
        if self.last:
            dt=frame-self.last[0];distance=math.dist(p,self.last[2]);reason=None
            if dt!=1 or t_ms<=self.last[1]:reason='id_gap_or_duplicate'
            elif distance>40 and distance>6*max(sorted(self.steps)[len(self.steps)//2] if self.steps else 0.,1e-6):reason='suspected_id_switch'
            if reason:
                self.fragmented=True
                self.diagnostic(dict(frame=self.last[0]+1 if reason=='id_gap_or_duplicate' else frame,track_id=self.tid,reason=reason,resumed_frame=frame))
                self.reset()
            else:self.steps=(self.steps+[distance])[-15:]
        if raw is None:
            self.fragmented=True;self.diagnostic(dict(frame=frame,track_id=self.tid,reason='ambiguous_overlap'))
            self.reset();self.last=(frame,t_ms,p);return
        target=raw
        if self.confirmed and signed_distance(p,self.polygons[self.confirmed])>=-self.hysteresis:target=self.confirmed
        if target and target!=self.confirmed and signed_distance(p,self.polygons[target])<self.hysteresis:
            self.candidate=None;self.run=0;self.last=(frame,t_ms,p);return
        if target==self.confirmed:
            self.anchor=p;self.candidate=None;self.run=0;self.minimum=1.
        else:
            if self.candidate==target:self.run+=1;self.minimum=min(self.minimum,confidence)
            else:self.candidate=target;self.run=1;self.minimum=confidence
            if self.run>=self.min_dwell and (self.confirmed is None or self.anchor is not None and math.dist(self.anchor,p)>=self.displacement):
                previous=self.confirmed
                if previous is not None:
                    event=dict(event_id=f'{self.tid}:{self.segment}:{frame}:boundary',kind='boundary',track_id=self.tid,segment_id=self.segment,
                        from_zone=previous,to_zone=target,frame=frame,timestamp_ms=t_ms,confidence=self.minimum,
                        confidence_kind='heuristic_detection_support_not_calibrated',from_position=self.anchor,to_position=p,via_outside=False)
                    self.emit(event)
                    if previous=='' and target and self.last_zone and self.last_zone!=target:
                        self.emit({**event,'event_id':f'{self.tid}:{self.segment}:{frame}:zone_path','kind':'zone_path','from_zone':self.last_zone,'via_outside':True})
                self.confirmed=target;self.anchor=p;self.candidate=None;self.run=0;self.minimum=1.
                if target:
                    if not self.path_length or self.path_last!=target or previous=='':
                        if not self.path_length:self.first_zone=target
                        self.path_last=target;self.path_length+=1
                    self.last_zone=target
        self.last=(frame,t_ms,p)

    def finish(self,frame_count):
        if self.last and self.last[0]<frame_count-1:self.diagnostic(dict(frame=self.last[0]+1,track_id=self.tid,reason='unobserved_destination'))
        if self.path_length:self.paths.append((self.first_zone,self.path_last,self.path_length))
        return self.paths[0][:2] if not self.fragmented and len(self.paths)==1 and self.paths[0][2]>=2 else None
