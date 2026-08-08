#!/usr/bin/env python3
"""Print event-level moments from NNLOJET profile histograms in a run dir.

usage: show_moments.py <dir> <PREFIX> <denom_tag> <prof_tag_base> <nmax> [label]
   e.g. show_moments.py ggh_moments H.GGH_MOMENTS.LO norm_born prof_yh 6 "|y_H|"
"""
import os
import sys


def val(path):
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        return float(line.split()[3])
    return None


def main():
    d, pref, den, base, nmax = sys.argv[1:6]
    label = sys.argv[6] if len(sys.argv) > 6 else base
    D = val(os.path.join(d, f"{pref}.{den}.s1.dat"))
    if D is None:
        print(f"  denominator {den} missing"); return
    print(f"  denominator {den} = {D:.6e}")
    for n in range(1, int(nmax) + 1):
        p = val(os.path.join(d, f"{pref}.{base}_{n}.s1.dat"))
        if p is None:
            print(f"    {base}_{n}: missing"); continue
        if D == 0:
            print(f"    <T_{n}({label})> = n/a (denominator 0)")
        else:
            print(f"    <T_{n}({label})> = {p/D:+.4f}")


if __name__ == "__main__":
    main()
