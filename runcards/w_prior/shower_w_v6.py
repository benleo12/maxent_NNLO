#!/usr/bin/env python3
"""[v6, referee round 2026-11-03: exact Born leptons (QED radiation off leptons and other non-QCD charged
particles switched off, so l_born = l_bare = the pre-FSR lepton), MECs at Pythia's default (on) unless --no-mec,
--shower-vars adds the non-singular cNS = +-2 variations; the npz records flags and w_shower_names.]
Shower W -> l nu LHE events (MG5 LO) with Pythia 8 and store per-event
Born-level W and lepton kinematics.

Mirrors shower_dy.py: LO LHE, power shower (pTmaxMatch = 2), parton level
(--no-hadron), optional --kt-hard override, optional Mrenna-Skands shower
variation weights.  The charged lepton is taken BORN-level: the highest-pT
final-state charged lepton with its own QED FSR photons added back
(shower_dy.find_born_leptons logic, one charge).  The neutrino is the
highest-pT final-state neutrino of the matching flavour.  The W is
l_born + nu, which is what NNLOJET's ptw / mln / yw evaluate.

Stored per event: l_born, l_bare, nu (px,py,pz,E), charge, pt_l, eta_l, y_l,
pt_nu, mt, pt_w, y_w, m_w, weight, in_fid (the ATLAS 7 TeV W selection
1612.03016: pT_l > 25, |eta_l| < 2.5, pT_nu > 25, m_T > 40), w_shower.
"""
import os, sys, argparse, time
import numpy as np
import pythia8


def hard_exit(code=0):
    try:
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(code)


def top_status(evt, i):
    """|status| of the top copy of particle i (follow mother1 while the mother has the same id)."""
    j = i
    while True:
        m = evt[j].mother1()
        if m > 0 and evt[m].id() == evt[j].id():
            j = m
        else:
            return abs(evt[j].status())


def find_w_decay(evt):
    """v6.1: (l_bare, l_born, nu, charge) of the hard-process charged lepton and neutrino (final particles whose top
    copy has |status| 23), or None.  With lepton QED off l_born = l_bare is the Born lepton.  The highest-pT final
    lepton is not always the W's: a heavy-flavour decay lepton can outrank a soft one."""
    il = inu = -1
    for i in range(1, evt.size()):
        p = evt[i]
        if not p.isFinal() or top_status(evt, i) != 23:
            continue
        if abs(p.id()) in (11, 13): il = i
        elif abs(p.id()) in (12, 14): inu = i
    if il < 0 or inu < 0:
        return None
    l = evt[il]; nu = evt[inu]
    best_l = np.array([l.px(), l.py(), l.pz(), l.e()]); best_nu = np.array([nu.px(), nu.py(), nu.pz(), nu.e()])
    charge = -1 if l.id() > 0 else +1      # PID +11/+13 = l- ; -11/-13 = l+
    return best_l, best_l.copy(), best_nu, charge


def kin(p4):
    px, py, pz, E = p4
    pt = float(np.hypot(px, py))
    y = float(0.5 * np.log((E + pz) / max(E - pz, 1e-30)))
    pabs = float(np.sqrt(px * px + py * py + pz * pz))
    eta = float(0.5 * np.log((pabs + pz) / max(pabs - pz, 1e-30)))
    phi = float(np.arctan2(py, px))
    return pt, y, eta, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lhe', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=0)
    ap.add_argument('--pT-lep', type=float, default=25.)
    ap.add_argument('--eta-lep', type=float, default=2.5)
    ap.add_argument('--pT-nu', type=float, default=25.)
    ap.add_argument('--mt-min', type=float, default=40.)
    ap.add_argument('--no-hadron', action='store_true')
    ap.add_argument('--no-mec', action='store_true', help='matrix-element corrections off (the Drell-Yan prior convention)')
    ap.add_argument('--skip', type=int, default=0, help='skip this many LHE events first (split one file over jobs)')
    ap.add_argument('--nevents', type=int, default=0, help='cap on LHE events read, failures included (chunks never overlap)')
    ap.add_argument('--kt-hard', type=float, default=None)
    ap.add_argument('--shower-vars', action='store_true')
    ap.add_argument('--seed', type=int, default=0, help='0 = Pythia default (no seed set, as shower_dy.py)')
    args = ap.parse_args()

    p = pythia8.Pythia()
    p.readString('Beams:frameType = 4')
    p.readString(f'Beams:LHEF = {args.lhe}')
    p.readString('SpaceShower:pTmaxMatch = 2')
    p.readString('TimeShower:pTmaxMatch = 2')
    if args.no_mec:
        p.readString('SpaceShower:MEcorrections = off')
        p.readString('TimeShower:MEcorrections = off')
    if args.skip:
        p.readString(f'Beams:nSkipLHEFatInit = {args.skip}')
    if args.no_hadron:
        p.readString('HadronLevel:all = off')
    if args.shower_vars:
        p.readString('UncertaintyBands:doVariations = on')
        p.readString('UncertaintyBands:List = {isrHi isr:muRfac=0.5, isrLo isr:muRfac=2.0, fsrHi fsr:muRfac=0.5, fsrLo fsr:muRfac=2.0, isrNSHi isr:cNS=2.0, isrNSLo isr:cNS=-2.0, fsrNSHi fsr:cNS=2.0, fsrNSLo fsr:cNS=-2.0}')
    if args.kt_hard is not None:
        p.readString(f'BeamRemnants:primordialKThard = {args.kt_hard}')
    if args.seed:
        p.readString('Random:setSeed = on'); p.readString(f'Random:seed = {args.seed}')
    p.readString('Beams:setProductionScalesFromLHEF = off')
    for q in ('TimeShower:QEDshowerByL = off', 'TimeShower:QEDshowerByOther = off', 'SpaceShower:QEDshowerByL = off'):
        assert p.readString(q), q
    p.readString('Print:quiet = on')
    p.init()

    keys = ('l_born', 'l_bare', 'nu', 'charge', 'pt_l', 'eta_l', 'y_l', 'pt_nu', 'mt',
            'pt_w', 'y_w', 'm_w', 'weight', 'in_fid', 'w_shower')
    out = {k: [] for k in keys}
    n_total = 0; n_fid = 0; sum_w_total = 0.0; sum_w_fid = 0.0; n_read = 0
    t0 = time.time()
    while True:
        if args.nevents and n_read >= args.nevents:
            break
        ok = p.next()
        if not ok and p.infoPython().atEndOfFile():
            break
        n_read += 1
        if not ok:
            continue
        info = p.infoPython()
        weight = info.weight()
        n_total += 1; sum_w_total += weight
        if args.max_events and n_total > args.max_events:
            n_total -= 1; break
        dec = find_w_decay(p.event)
        if dec is None:
            continue
        l_bare, l_born, nu, charge = dec
        pt_l, y_l, eta_l, phi_l = kin(l_born)
        pt_nu, _, _, phi_nu = kin(nu)
        dphi = np.arctan2(np.sin(phi_l - phi_nu), np.cos(phi_l - phi_nu))
        mt = float(np.sqrt(max(2.0 * pt_l * pt_nu * (1.0 - np.cos(dphi)), 0.0)))
        w4 = l_born + nu
        pt_w, y_w, _, _ = kin(w4)
        m_w = float(np.sqrt(max(w4[3] ** 2 - w4[0] ** 2 - w4[1] ** 2 - w4[2] ** 2, 0.0)))
        in_fid = (pt_l > args.pT_lep and abs(eta_l) < args.eta_lep and pt_nu > args.pT_nu and mt > args.mt_min)
        out['l_born'].append(l_born); out['l_bare'].append(l_bare); out['nu'].append(nu)
        out['charge'].append(charge); out['pt_l'].append(pt_l); out['eta_l'].append(eta_l); out['y_l'].append(y_l)
        out['pt_nu'].append(pt_nu); out['mt'].append(mt); out['pt_w'].append(pt_w); out['y_w'].append(y_w)
        out['m_w'].append(m_w); out['weight'].append(weight); out['in_fid'].append(in_fid)
        if args.shower_vars:
            out['w_shower'].append([info.weightValueByIndex(i) * weight / max(info.weightValueByIndex(0), 1e-300)
                                    for i in range(info.numberOfWeights())])
        if in_fid:
            n_fid += 1; sum_w_fid += weight
        if n_total % 50000 == 0:
            print(f'  total={n_total:,}  fid={n_fid:,}  rate={n_total/max(time.time()-t0,1e-6):.0f} ev/s', flush=True)

    arrs = {k: np.asarray(v) for k, v in out.items() if len(v)}
    sig_lhe = sum(info.sigmaLHEF(i) for i in range(info.nProcessesLHEF()))
    arrs['flags'] = np.array([f'no_mec={args.no_mec}', f'kt_hard={args.kt_hard}', f'no_hadron={args.no_hadron}', 'QEDshowerByL=off', 'QEDshowerByOther=off', f'skip={args.skip}', f'nevents={args.nevents}', f'seed={args.seed}'])
    if args.shower_vars: arrs['w_shower_names'] = np.array([info.weightNameByIndex(i) for i in range(info.numberOfWeights())])
    np.savez_compressed(args.out, n_fail=np.int64(n_read - n_total), n_total=np.int64(n_total), n_read=np.int64(n_read), sum_w_total=np.float64(sum_w_total),
                        sum_w_fid=np.float64(sum_w_fid), sigma_lhe_pb=np.float64(sig_lhe),
                        sigma_gen_pb=np.float64(info.sigmaGen() * 1e9), **arrs)
    print(f'DONE: {n_total:,} showered, {n_fid:,} fiducial ({100*n_fid/max(n_total,1):.1f}%), '
          f'sigma_LHE {sig_lhe:.3f} pb -> {args.out}', flush=True)
    hard_exit(0)


if __name__ == '__main__':
    main()
