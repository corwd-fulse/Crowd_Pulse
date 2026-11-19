"""Original-image pixel geometry shared by revision validation and replay."""
import math
import re

EPS=1e-7
def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def on_segment(a,b,p):
    return abs(cross(a,b,p))<EPS and min(a[0],b[0])-EPS<=p[0]<=max(a[0],b[0])+EPS and min(a[1],b[1])-EPS<=p[1]<=max(a[1],b[1])+EPS
def intersects(a,b,c,d):
    return (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on_segment(a,b,c),on_segment(a,b,d),on_segment(c,d,a),on_segment(c,d,b)))
def inside(point,polygon):
    result=False
    for i,a in enumerate(polygon):
        b=polygon[(i+1)%len(polygon)]
        if on_segment(a,b,point):return True
        if (a[1]>point[1])!=(b[1]>point[1]) and point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]:result=not result
    return result
def signed_distance(point,polygon):
    distances=[]
    for i,a in enumerate(polygon):
        b=polygon[(i+1)%len(polygon)];dx,dy=b[0]-a[0],b[1]-a[1]
        length=dx*dx+dy*dy
        t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length)) if length else 0.
        distances.append(math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy))
    distance=min(distances)
    return distance if inside(point,polygon) else -distance
def validate_disjoint(zones):
    for index,z in enumerate(zones):
        p=z['polygon'];n=len(p)
        if z.get('color') is not None and (not isinstance(z['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',z['color'])):raise ValueError('Invalid color')
        if 'visible' in z and type(z['visible']) is not bool:raise ValueError('Invalid visibility')
        for i,a in enumerate(p):
            b=p[(i+1)%n];c=p[(i+2)%n]
            if math.dist(a,b)<EPS or (abs(cross(a,b,c))<EPS and (on_segment(a,b,c) or on_segment(b,c,a))):raise ValueError('Repeated/backtracking edge')
            for j in range(i+1,n):
                if j==i+1 or (i==0 and j==n-1):continue
                if intersects(a,b,p[j],p[(j+1)%n]):raise ValueError('Self intersection')
        for other in zones[:index]:
            q=other['polygon']
            if any(intersects(a,p[(i+1)%n],b,q[(j+1)%len(q)]) for i,a in enumerate(p) for j,b in enumerate(q)) or inside(p[0],q) or inside(q[0],p):
                raise ValueError('Zones overlap or touch; leave an unassigned gap')
