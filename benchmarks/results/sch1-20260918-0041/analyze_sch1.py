import xml.etree.ElementTree as ET, re, statistics, sys
F=f"benchmarks/results/{sys.argv[1]}/sch1-gpu-intervals.xml"
tree=ET.parse(F); root=tree.getroot()
# id -> text (resolve refs)
id2text={}
for el in root.iter():
    if el.get('id') is not None:
        id2text[el.get('id')]=el.text if el.text is not None else ''
# id -> pid
id2pid={}
for el in root.iter():
    if el.tag=='process' and el.get('id') is not None:
        fmt=el.get('fmt','')
        m=re.search(r'\((\d+)\)', fmt)
        if m: id2pid[el.get('id')]=int(m.group(1))
        else:
            m2=re.match(r'(\d+)$', fmt.strip())
            if m2: id2pid[el.get('id')]=int(m2.group(1))
def val(el):
    if el is None: return None
    if el.text is not None: return el.text
    r=el.get('ref')
    return id2text.get(r) if r else None
rows=root.findall('.//row')
# server pid
server_pid=max(id2pid.values(), key=lambda p: 0)  # placeholder
from collections import Counter
intervals=[]  # (start_ns, dur_ns, pid)
for r in rows:
    st=r.find('start-time'); du=r.find('duration'); pr=r.find('process')
    if st is None or du is None or pr is None: continue
    s=val(st); d=val(du)
    if s is None or d is None: continue
    try: s=int(s); d=int(d)
    except: continue
    pid=id2pid.get(pr.get('ref') or pr.get('id'))
    intervals.append((s,d,pid))
c=Counter(p for _,_,p in intervals)
print("rows by pid:", dict(c), file=sys.stderr)
