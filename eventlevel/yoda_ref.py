"""Minimal YODA v2 reader for the REF scatters Rivet ships."""
import numpy as np, re

def read(path, name):
    lines = open(path).read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("BEGIN YODA") and l.rstrip().endswith(name))
    i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].startswith("END YODA"))
    rows = []
    for l in lines[i0 + 1:i1]:
        l = l.strip()
        if not l or l.startswith("#") or "=" in l.split()[0] or l[0].isalpha():
            continue
        p = l.split()
        try: rows.append([float(x) for x in p])
        except ValueError: continue
    a = np.array(rows)
    return dict(x=a[:, 0], exl=a[:, 1], exh=a[:, 2], y=a[:, 3], eyl=a[:, 4], eyh=a[:, 5],
                lo=a[:, 0] - a[:, 1], hi=a[:, 0] + a[:, 2])
