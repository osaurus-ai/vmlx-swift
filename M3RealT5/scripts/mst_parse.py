"""Summarize an xctrace Metal System Trace export: GPU intervals by label/channel, compiler activity count."""
import collections, subprocess, sys, xml.etree.ElementTree as ET
trace = sys.argv[1]
def table(schema):
    x = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
                        f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'], capture_output=True, text=True).stdout
    root = ET.fromstring(x)
    ids = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}
    rows = []
    for row in root.iter("row"):
        r = []
        for c in row:
            if "ref" in c.attrib:
                c = ids[c.attrib["ref"]]
            r.append((c.tag, c.attrib.get("fmt", c.text), c.text))
        rows.append(r)
    return rows
rows = table("metal-gpu-intervals")
agg = collections.defaultdict(list)
for r in rows:
    d = dict((t, (f, v)) for t, f, v in r)
    dur = int(d["duration"][1]) / 1e6
    label = d.get("formatted-label", ("?", ""))[0]
    chan = d.get("gpu-channel-name", ("?", ""))[0]
    depth = d.get("metal-nesting-level", ("?", ""))[0]
    agg[(chan, depth, label[:60])].append(dur)
for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:15]:
    v.sort()
    print(f"{k}: n={len(v)} total={sum(v):.1f} ms median={v[len(v)//2]:.4f} ms max={v[-1]:.3f}")
comp = table("graphics-compiler-activity-intervals")
print("compiler activity intervals:", len(comp), "total ms", sum(int(dict((t,(f,v)) for t,f,v in r)["duration"][1]) for r in comp) / 1e6)
for r in comp[:8]:
    d = dict((t, (f, v)) for t, f, v in r); print("  ", d["start-time"][0], d["duration"][0], d.get("formatted-label", ("",""))[0][:90])
