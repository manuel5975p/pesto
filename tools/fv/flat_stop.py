#!/usr/bin/env python3
"""flat_stop.py CSV TOL SPAN TMIN -- CMD...: run CMD, SIGTERM it when the means of cd over the last two SPAN windows agree to TOL (t >= TMIN). Prints the mean."""
import sys, subprocess, time, signal
i=sys.argv.index('--'); csv,tol,span,tmin=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]); cmd=sys.argv[i+1:]
p=subprocess.Popen(cmd,stdout=subprocess.DEVNULL)
def rows():
    try: L=[l.split(',') for l in open(csv) if not l.startswith('#')]
    except OSError: return []
    if len(L)<3 or 'cd' not in L[0]: return []
    it,ic=L[0].index('t'),L[0].index('cd'); o=[]
    for r in L[1:]:
        try: o.append((float(r[it]),float(r[ic])))
        except (ValueError,IndexError): pass
    return o
def means(R):
    e=R[-1][0]; a=[c for t,c in R if e-span<=t]; b=[c for t,c in R if e-2*span<=t<e-span]
    return (sum(a)/len(a), sum(b)/len(b)) if a and b else None
while p.poll() is None:
    time.sleep(2)
    R=rows()
    if not R or R[-1][0]<max(tmin,2*span): continue
    m=means(R)
    if m and abs(m[0]-m[1])<tol: p.send_signal(signal.SIGTERM); break
rc=p.wait(); R=rows()
if R:
    m=means(R); print(f"RESULT rc {rc} t {R[-1][0]:.5g} cdlast {R[-1][1]:.6f} mean {m[0]:.6f} prev {m[1]:.6f}")
sys.exit(0)
