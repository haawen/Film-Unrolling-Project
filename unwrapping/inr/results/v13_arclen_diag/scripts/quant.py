"""Separate the ONCE-PER-TURN component (what arc columns fix) from everything
else, per winding block, for v12 vs the simulated v13."""
import math, os, numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = os.path.dirname(os.path.abspath(__file__)); TWO_PI = 2*math.pi; EPS = math.radians(0.3)
prof = np.load(os.path.join(SP, "v12_profile.npy"))
d = np.load(r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v11\matched_walks.npz", allow_pickle=True)
cx = np.asarray(d["cx_a"],float)[:,None]; cy = np.asarray(d["cy_a"],float)[:,None]
paths = d["paths"]; n=len(paths); nw=len(paths[0])
Ks=[paths[0][w].shape[0] for w in range(nw)]
Ms=[int(round(2*EPS*Ks[i]/(TWO_PI-2*EPS))) for i in range(nw-1)]+[0]
starts=np.cumsum([0]+[Ks[i]+Ms[i] for i in range(nw)])
out=[]; b13=[0]
for wi in range(nw):
    K=Ks[wi]; blk=prof[starts[wi]:starts[wi]+K]; phi=np.linspace(EPS,TWO_PI-EPS,K)
    R=np.hypot(np.stack([paths[a][wi][:,0] for a in range(n)])-cx,
               np.stack([paths[a][wi][:,1] for a in range(n)])-cy)
    rm=np.median(R,axis=0); ds=np.hypot(rm,np.gradient(rm,phi))
    s=np.concatenate([[0.],np.cumsum(.5*(ds[1:]+ds[:-1])*np.diff(phi))])
    K2=max(2,int(round(s[-1]))); pn=np.interp(np.linspace(0,s[-1],K2),s,phi)
    out.append(np.interp(pn,phi,blk)); b13.append(b13[-1]+K2)
    if Ms[wi]: out.append(prof[starts[wi]+K:starts[wi]+K+Ms[wi]]); b13[-1]+=Ms[wi]
prof13=np.concatenate(out)
def lp(p,f0=1/903.6,bw=.28,sm=400):
    p=p-gaussian_filter1d(p,1500); F=np.fft.rfft(p); f=np.fft.rfftfreq(len(p))
    F*=np.exp(-.5*((f-f0)/(bw*f0))**2); band=np.fft.irfft(F,n=len(p))
    return TWO_PI/np.maximum(gaussian_filter1d(np.gradient(np.unwrap(np.angle(hilbert(band)))),sm),1e-9)
p12,p13=lp(prof),lp(prof13)
def per_block(pit, bounds, tag):
    amps=[]
    for i in range(2,nw-2):                       # skip the film-end fragments
        lo,hi=bounds[i],bounds[i+1]
        y=pit[lo+300:hi-300]
        if len(y)<1500: continue
        y=y/np.median(y)-1.0
        t=np.linspace(0,TWO_PI,len(y))            # ONE turn spans the block
        A=math.hypot(2*np.mean(y*np.cos(t)), 2*np.mean(y*np.sin(t)))   # 1/turn amp
        res=y-(np.mean(y*np.cos(t))*2*np.cos(t)+np.mean(y*np.sin(t))*2*np.sin(t))
        amps.append((100*A, 100*res.std()))
    a=np.array(amps)
    print(f"  {tag:22s} once-per-turn amplitude {a[:,0].mean():5.2f}%   "
          f"everything else (std) {a[:,1].mean():5.2f}%")
    return a
b12=np.cumsum([0]+[Ks[i]+Ms[i] for i in range(nw)])
print("per-winding decomposition of the local-pitch modulation:")
A12=per_block(p12,b12,"v12 uniform-azimuth"); A13=per_block(p13,np.array(b13),"v13 arc-length")
print(f"\n  => once-per-turn breathing reduced {A12[:,0].mean()/A13[:,0].mean():.1f}x "
      f"({A12[:,0].mean():.2f}% -> {A13[:,0].mean():.2f}%); "
      f"broadband residual {A12[:,1].mean():.2f}% -> {A13[:,1].mean():.2f}% (unchanged, "
      f"= real film + estimator noise)")
# do the leftover spikes sit at winding joins?
sp=np.where(np.abs(p13/np.median(p13)-1)>0.06)[0]
if len(sp):
    grp=np.split(sp,np.where(np.diff(sp)>2000)[0]+1)
    print("\n  residual spikes at cols:", [int(np.mean(g)) for g in grp])
    print("  winding-block joins at :", [int(v) for v in b13[1:-1]][:40])
    near=[min(abs(int(np.mean(g))-v) for v in b13) for g in grp]
    print(f"  distance to nearest join: {near}  -> "
          f"{'JOIN artifacts' if max(near)<3000 else 'not all at joins'}")
fig,ax=plt.subplots(figsize=(9,4))
x=np.arange(len(A12))
ax.bar(x-0.2,A12[:,0],0.4,label=f"v12 uniform-azimuth (mean {A12[:,0].mean():.2f}%)")
ax.bar(x+0.2,A13[:,0],0.4,label=f"v13 arc-length (mean {A13[:,0].mean():.2f}%)")
ax.set_xlabel("winding"); ax.set_ylabel("once-per-turn pitch modulation (%)")
ax.set_title("The breathing this fixes, per winding"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(SP,"quant.png"),dpi=110); print("saved quant.png")
