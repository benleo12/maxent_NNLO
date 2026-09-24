#!/usr/bin/env python3
r"""Can the two CMS 13 TeV Drell-Yan measurements James suggested be overlaid on
our samples?  Answer, measured here: not without a fixed-order run at their cuts.

  CMS_2019_I1753680 (arXiv:1909.04133)  |y_ll|, pT, phi*   table d32-x01-y01 is
      the NORMALISED |y_ll|, which is what our shape comparisons use.
      Selection: pT_l > 25, |eta_l| < 2.4, 76.1876 < m_ll < 106.1876, leptons
      dressed with dR = 0.1.
  CMS_2018_I1711625 (arXiv:1812.10529)  m_ll, 15-3000 GeV.  Fiducial muon table
      d05-x01-y01.  Selection: leading pT > 22 (mu) / 30 (e), SUBLEADING pT > 10,
      |eta_l| < 2.4 (mu) / 2.5 (e), dressed dR = 0.1.
  Ours (ATLAS 1912.02844): 66 < m_ll < 116, pT_l > 27 both, |eta_l| < 2.5.

Rivet 3.1.10 ships both reference sets, so no download is needed:
  /opt/homebrew/Cellar/rivet/3.1.10/share/Rivet/<ANALYSIS>.yoda

Run this to reproduce the numbers in the message to James.
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import yoda_ref
RD = os.environ.get("RIVET_DATA", "/opt/homebrew/Cellar/rivet/3.1.10/share/Rivet")
PRIOR = os.path.join(HERE, "dy_psLO_born_1.npz")   # parent: ALL showered events, no fiducial cut


def kinematics(z, dressed=True):
    kp, km = ("l_plus_born", "l_minus_born") if dressed else ("l_plus", "l_minus")
    lp, lm = np.asarray(z[kp], float), np.asarray(z[km], float)
    px, py, pz, E = (lp[:, i] + lm[:, i] for i in range(4))
    mll = np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0.0))
    y = np.abs(0.5 * np.log(np.clip((E + pz) / np.maximum(E - pz, 1e-12), 1e-12, None)))
    pt = lambda v: np.hypot(v[:, 0], v[:, 1])
    def eta(v):
        p = np.sqrt((v[:, :3] ** 2).sum(1))
        return np.arctanh(np.clip(v[:, 2] / np.maximum(p, 1e-12), -1 + 1e-12, 1 - 1e-12))
    return dict(mll=mll, y=y, ptsub=np.minimum(pt(lp), pt(lm)), ptlead=np.maximum(pt(lp), pt(lm)),
                emax=np.maximum(np.abs(eta(lp)), np.abs(eta(lm))))


def main():
    z = np.load(PRIOR); w = np.asarray(z["weight"], float); k = kinematics(z)
    A = (k["mll"] >= 66) & (k["mll"] <= 116) & (k["ptsub"] > 27) & (k["emax"] < 2.5)
    C = (k["mll"] >= 76.1876) & (k["mll"] <= 106.1876) & (k["ptsub"] > 25) & (k["emax"] < 2.4)
    D = yoda_ref.read(os.path.join(RD, "CMS_2019_I1753680.yoda"), "d32-x01-y01")
    e = np.concatenate([D["lo"][:1], D["hi"]]); bw = np.diff(e)
    def nd(m):
        h, _ = np.histogram(k["y"][m], e, weights=w[m]); h = h / bw; return h / (h * bw).sum()
    ref = nd(A)
    print("Normalised |y_ll|: effect of ONE cut change at a time, ATLAS -> CMS 2019")
    for name, alt in (("mass window", (k["mll"] >= 76.1876) & (k["mll"] <= 106.1876) & (k["ptsub"] > 27) & (k["emax"] < 2.5)),
                      ("lepton pT 27->25", (k["mll"] >= 66) & (k["mll"] <= 116) & (k["ptsub"] > 25) & (k["emax"] < 2.5)),
                      ("lepton |eta| 2.5->2.4", (k["mll"] >= 66) & (k["mll"] <= 116) & (k["ptsub"] > 27) & (k["emax"] < 2.4))):
        r = np.abs(nd(alt) / ref - 1)
        print(f"  {name:22s} median {100*np.median(r):6.2f}%   max {100*r.max():6.2f}%")
    r = np.abs(nd(C) / ref - 1)
    print(f"  {'all three':22s} median {100*np.median(r):6.2f}%   max {100*r.max():6.2f}%")
    print(f"  CMS data errors                {100*np.min(D['eyh']/np.abs(D['y'])):.2f}-{100*np.max(D['eyh']/np.abs(D['y'])):.2f}%")
    both = A & C
    rb = np.abs(nd(both) / nd(C) - 1)
    print(f"\nUsing only events in BOTH selections (the ones we have MaxEnt weights for):")
    print(f"  {100*w[both].sum()/w[C].sum():.2f}% of the CMS weight; the missing 25 < pT < 27 GeV events")
    print(f"  bias the normalised shape by median {100*np.median(rb):.2f}%, max {100*rb.max():.2f}%")
    D2 = yoda_ref.read(os.path.join(RD, "CMS_2018_I1711625.yoda"), "d05-x01-y01")
    e2 = np.concatenate([D2["lo"][:1], D2["hi"]])
    inwin = (e2[:-1] >= 66) & (e2[1:] <= 116)
    print(f"\nCMS_2018 m_ll: {len(D2['y'])} bins over {e2[0]:g}-{e2[-1]:g} GeV, of which {inwin.sum()} lie inside")
    print(f"  our constrained window 66-116 GeV; its subleading lepton cut is 10 GeV against our 27.")


if __name__ == "__main__":
    main()
