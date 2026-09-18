import xml.etree.ElementTree as ET, re, statistics, sys
RUNID=sys.argv[1]
OUTDIR=f"benchmarks/results/{RUNID}"

# --- 1. server GPU intervals (start_ns, dur_ns) trace-time, sorted ---
tree=ET.parse(f"{OUTDIR}/sch1-gpu-intervals.xml"); root=tree.getroot()
id2text={el.get('id'): (el.text if el.text is not None else '') for el in root.iter() if el.get('id') is not None}
id2pid={}
for el in root.iter():
    if el.tag=='process' and el.get('id') is not None:
        fmt=el.get('fmt',''); m=re.search(r'\((\d+)\)', fmt)
        if m: id2pid[el.get('id')]=int(m.group(1))
        else:
            m2=re.match(r'(\d+)$', fmt.strip())
            if m2: id2pid[el.get('id')]=int(m2.group(1))
def val(el):
    if el is None: return None
    if el.text is not None: return el.text
    return id2text.get(el.get('ref'))
iv=[]
for r in root.findall('.//row'):
    st=r.find('start-time'); du=r.find('duration'); pr=r.find('process')
    if st is None or du is None or pr is None: continue
    s=val(st); d=val(du)
    if s is None or d is None: continue
    try: s=int(s); d=int(d)
    except: continue
    if id2pid.get(pr.get('ref') or pr.get('id'))==6733:
        iv.append((s,d))
iv.sort()
print(f"server GPU intervals: {len(iv)}", file=sys.stderr)

# --- 2. mtp-anchor eval windows (mach uptime) ---
anchors={}
for line in open(f"{OUTDIR}/sch1-mtp-trace.log"):
    if not line.startswith("mtp-anchor:"): continue
    d={}
    for kv in line.strip().split()[1:]:
        if '=' in kv:
            k,v=kv.split('='); d[k]=int(v)
    anchors[d['round']]=d
# --- 3. mtp-trace phase us ---
phases={}
for line in open(f"{OUTDIR}/sch1-mtp-trace.log"):
    if not line.startswith("mtp-trace: round="): continue
    d={}
    for kv in line.strip().split()[1:]:
        if kv.endswith('_us'):
            k,v=kv.split('='); d[k[:-3]]=int(v)
        elif kv.startswith('round='):
            d['round']=int(kv.split('=')[1])
    phases[d['round']]=d

T0=984400672679458
# eval windows in mach: (vb, ed)
evw=[(a['t_verify_built'], a['t_eval_done']) for a in anchors.values() if 't_verify_built' in a]

def merge_windows(win):
    win=sorted(win); out=[]
    for s,e in win:
        if out and s<=out[-1][1]: out[-1]=(out[-1][0], max(out[-1][1],e))
        else: out.append((s,e))
    return out
def busy_in_union(delta_ns):
    # eval windows shifted to trace-time
    win=[(vb-(T0+delta_ns), ed-(T0+delta_ns)) for vb,ed in evw]
    uni=merge_windows(win)
    total=0; j=0
    for (s,d) in iv:
        e=s+d
        while j<len(uni) and uni[j][1]<=s: j+=1
        k=j
        while k<len(uni) and uni[k][0]<e:
            total+=min(e,uni[k][1])-max(s,uni[k][0]); k+=1
    return total
# coarse search
best=(0,0)
for dm in range(-40,41):
    b=busy_in_union(dm*1e6)
    if b>best[1]: best=(dm,b)
# fine
for dm in range(best[0]*10-10, best[0]*10+11):
    b=busy_in_union(dm*1e5)
    if b>best[1]: best=(dm/10,b)
print(f"calibrated delta={best[0]}ms busy={best[1]/1e6:.1f}ms", file=sys.stderr)
import json
json.dump({"delta_ms":best[0],"busy_total_ms":best[1]/1e6}, open(f"/tmp/sch-calib.json","w"))
