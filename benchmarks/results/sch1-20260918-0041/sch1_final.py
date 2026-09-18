import xml.etree.ElementTree as ET, re, statistics, sys
RUNID=open('/tmp/sch-runid.txt').read().strip()
OUTDIR=f"benchmarks/results/{RUNID}"
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
# parse anchors + phases
anchors={}; phases={}
for line in open(f"{OUTDIR}/sch1-mtp-trace.log"):
    if line.startswith("mtp-anchor:"):
        d={}
        for kv in line.strip().split()[1:]:
            if '=' in kv:
                k,v=kv.split('='); d[k]=int(v)
        anchors[d['round']]=d
    elif line.startswith("mtp-trace: round="):
        round_num=None; d={}
        for kv in line.strip().split()[1:]:
            if '=' not in kv: continue
            k,v=kv.split('=')
            if k=='round': round_num=int(v)
            elif k.endswith('_us'): d[k[:-3]]=int(v)
        phases[round_num]=d
T0=984400672679458
# per-round GPU busy in eval window, calibrate delta per round then median
def busy_in(delta_ns, lo, hi):
    # sum iv overlap with [lo,hi] (trace-time)
    total=0
    for (s,d) in iv:
        e=s+d
        if e<=lo: continue
        if s>=hi: break
        total+=min(e,hi)-max(s,lo)
    return total
# calibrate: for each round, find delta maximizing busy in its eval window
deltas=[]
for rn in sorted(anchors):
    a=anchors[rn]
    if 't_verify_built' not in a: continue
    vb,ed=a['t_verify_built'],a['t_eval_done']
    best=(0,0)
    for dm in range(30,50):  # expect ~40ms
        b=busy_in(dm*1e6, vb-(T0+dm*1e6), ed-(T0+dm*1e6))
        if b>best[1]: best=(dm,b)
    deltas.append(best[0])
delta_med=statistics.median(deltas)
print(f"per-round delta: median={delta_med}ms min={min(deltas)} max={max(deltas)} n={len(deltas)}")
delta=delta_med*1e6
# per-round ledger (steady-state: exclude round 1 which is cold/prefill)
import collections
rows=[]
for rn in sorted(anchors):
    a=anchors[rn]; p=phases.get(rn, {})
    if 't_verify_built' not in a or 'eval_wall' not in p: continue
    vb,ed=a['t_verify_built'],a['t_eval_done']
    busy=busy_in(delta, vb-(T0+delta), ed-(T0+delta))  # ns
    eval_wall_ns=p['eval_wall']*1000
    i_kernel=busy
    ii_sync=eval_wall_ns-busy
    iii_host=p.get('verify_build',0)*1000 + p.get('readout',0)*1000
    iv_struct=p.get('commit',0)*1000 + p.get('upkeep',0)*1000
    round_us=p.get('round',0)
    rows.append(dict(rn=rn, eval_wall=eval_wall_ns/1e6, i_kernel=i_kernel/1e6, ii_sync=ii_sync/1e6,
                     iii_host=iii_host/1e6, iv_struct=iv_struct/1e6, round_ms=round_us/1e3,
                     i2iv=ii_sync/1e6+iv_struct/1e6))
# steady-state = rounds 5..N
ss=[r for r in rows if r['rn']>=5]
def med(k): return statistics.median(r[k] for r in ss)
print(f"\n=== steady-state ledger (rounds 5..{ss[-1]['rn']}, n={len(ss)}) ===")
print(f"round_ms        med={med('round_ms'):.2f}")
print(f"eval_wall       med={med('eval_wall'):.2f} ms")
print(f"(i) kernel exec med={med('i_kernel'):.2f} ms  (GPU busy in eval)")
print(f"(ii) sync/idle  med={med('ii_sync'):.3f} ms  (eval - GPU busy)")
print(f"(iii) host build/read med={med('iii_host'):.3f} ms  (verify_build+readout)")
print(f"(iv) round-struct   med={med('iv_struct'):.3f} ms  (commit+upkeep)")
print(f"")
print(f"(ii)+(iv) = {med('ii_sync')+med('iv_struct'):.3f} ms  -> KILL SWITCH < 8ms? {'YES STOP' if med('ii_sync')+med('iv_struct')<8 else 'NO, proceed'}")
print(f"GPU util in eval = {med('i_kernel')/med('eval_wall')*100:.1f}%")
import json
json.dump({"delta_ms":delta_med,"steady":ss}, open(f"{OUTDIR}/sch1-ledger.json","w"))
