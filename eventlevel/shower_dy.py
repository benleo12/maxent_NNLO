#!/Users/user/miniconda3/envs/nnloreweight/bin/python3
"""Shower DY LHE events with Pythia8, save per-event lepton kinematics.

For LO LHE: standard shower (pTmaxMatch=2, kinematic limit).
For NLO LHE: MC@NLO matching mode (pTmaxMatch=1, uses SCALUP per event).

Final state: two charged leptons (e+/e- or mu+/mu-). After Pythia FSR, take the
status-stable lepton pair (post-photon-FSR) — for unfolded reweighting we want the
"dressed" leptons including collinear photons within a small cone, but for this
toy validation we just take the highest-pT same-flavor opposite-charge pair.

Outputs NPZ with: l_plus (Nx4), l_minus (Nx4), mll, pT_ll, y_ll, dy_pm, dphi_pm,
                  weight, in_fo (FO cut flag), n_total, sum_w_total, sum_w_fo, sigma_FO.
"""
import os, sys, argparse, time
import numpy as np
import pythia8


def hard_exit(code=0):
    try:
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(code)


def find_dressed_leptons(evt):
    """Return (l_plus_4mom, l_minus_4mom) as numpy arrays (px,py,pz,E), or None.
    Picks final-state e+/e- or mu+/mu- with highest pT per charge sign.
    """
    best_plus = None; best_plus_pt = 0.0
    best_minus = None; best_minus_pt = 0.0
    for i in range(1, evt.size()):
        p = evt[i]
        if not p.isFinal():
            continue
        pid = p.id()
        if abs(pid) not in (11, 13):
            continue
        pt = p.pT()
        # MG5/Pythia convention: PID +11/+13 = electron/muon (lepton, q=-1)
        if pid > 0 and pt > best_minus_pt:
            best_minus = np.array([p.px(), p.py(), p.pz(), p.e()]); best_minus_pt = pt
        elif pid < 0 and pt > best_plus_pt:
            best_plus = np.array([p.px(), p.py(), p.pz(), p.e()]); best_plus_pt = pt
    if best_plus is None or best_minus is None:
        return None
    return best_plus, best_minus


def find_born_leptons(evt):
    """Born-level leptons: the highest-pT final-state lepton of each charge PLUS
    every final-state photon whose mother chain passes through a charged lepton
    (i.e. its own QED FSR added back).  This is the post-ISR-recoil, pre-FSR
    lepton -- the convention of Born-level unfolded tables and of a QCD-only
    fixed-order calculation.  NOT the |status|==23 entries: those predate the
    ISR recoil in Pythia's record and have pT_ll identically zero.
    Returns (l_plus, l_minus) as (px,py,pz,E) arrays, or None.
    """
    leps = find_dressed_leptons(evt)
    if leps is None:
        return None
    l_plus, l_minus = leps[0].copy(), leps[1].copy()
    n = evt.size()
    for i in range(1, n):
        g = evt[i]
        if g.id() != 22 or not g.isFinal():
            continue
        j = g.mother1(); steps = 0; sign = 0
        while j > 0 and steps < 60:
            q = evt[j]; a = abs(q.id())
            if a in (11, 13):
                sign = 1 if q.id() > 0 else -1   # +: l- (pid>0), -: l+ (pid<0)
                break
            j = q.mother1(); steps += 1
        if sign == 0:
            continue
        v = np.array([g.px(), g.py(), g.pz(), g.e()])
        if sign > 0: l_minus += v
        else:        l_plus += v
    return l_plus, l_minus


def to_ptyphi(p4):
    px, py, pz, E = p4
    pt = np.sqrt(px*px + py*py)
    em = max(E - pz, 1e-30)
    y = 0.5 * np.log((E + pz) / em)
    phi = np.arctan2(py, px)
    return pt, y, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lhe', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=0)
    ap.add_argument('--mcatnlo', action='store_true',
                    help='enable MC@NLO matching (pTmaxMatch=1, SCALUP-driven)')
    ap.add_argument('--mlm', action='store_true',
                    help='enable MLM jet-matching merging (for LO+jets samples)')
    ap.add_argument('--mlm-qcut', type=float, default=20.,
                    help='MLM matching scale in GeV (must match LHE xqcut)')
    ap.add_argument('--mlm-njetmax', type=int, default=1,
                    help='MLM nJetMax (highest multiplicity in LHE)')
    ap.add_argument('--powheg', action='store_true',
                    help='enable POWHEG matching (pTmaxMatch=2 + POWHEG vetoes)')
    ap.add_argument('--pT-lep', type=float, default=25.)
    ap.add_argument('--eta-lep', type=float, default=2.4)
    ap.add_argument('--mll-min', type=float, default=60.)
    ap.add_argument('--mll-max', type=float, default=2000.)
    ap.add_argument('--no-hadron', action='store_true', help='disable hadronization')
    ap.add_argument('--kt-hard', type=float, default=None, help='override BeamRemnants:primordialKThard')
    ap.add_argument('--shower-vars', action='store_true', help='Pythia ISR/FSR muR variation weights (Mrenna-Skands)')
    ap.add_argument('--rwl', action='store_true', help='carry LHEF3 <weights> scale variations through (DIY pre-scan)')
    args = ap.parse_args()

    p = pythia8.Pythia()
    p.readString('Beams:frameType = 4')
    p.readString(f'Beams:LHEF = {args.lhe}')
    if args.mcatnlo:
        p.readString('SpaceShower:pTmaxMatch = 1')
        p.readString('TimeShower:pTmaxMatch = 1')
    elif args.powheg:
        # POWHEG matching, EXACT settings of the official Pythia8 example main31.cc:
        # POWHEG matching in a PYTHON-ONLY shower: the main31.cc power-shower+PowhegHooks
        # veto needs the C++ UserHook attached (setUserHooksPtr), which the Pythia8 Python
        # binding does not expose. Without the hook, pTmaxMatch=2 is an UNVETOED power
        # shower and gives an unphysically hard recoil. The correct hook-free matching is
        # the SCALUP-limited shower: pTmaxMatch=1 caps each event's shower at the POWHEG
        # emission scale, which reproduces the veto to good approximation.
        p.readString('SpaceShower:pTmaxMatch = 1')
        p.readString('TimeShower:pTmaxMatch = 1')
    elif args.mlm:
        # MLM merging: Pythia applies jet-matching veto using LHE xqcut info
        p.readString('JetMatching:merge = on')
        p.readString('JetMatching:scheme = 1')          # 1 = MadGraph-style MLM
        p.readString('JetMatching:setMad = on')         # read xqcut from LHE
        p.readString(f'JetMatching:qCut = {args.mlm_qcut}')
        p.readString(f'JetMatching:nJetMax = {args.mlm_njetmax}')
        p.readString('JetMatching:nQmatch = 5')
        p.readString('JetMatching:jetAlgorithm = 2')    # kT algorithm
        p.readString('JetMatching:coneRadius = 1.0')
        # Standard shower settings
        p.readString('SpaceShower:pTmaxMatch = 2')
        p.readString('TimeShower:pTmaxMatch = 2')
    else:
        p.readString('SpaceShower:pTmaxMatch = 2')
        p.readString('TimeShower:pTmaxMatch = 2')
    if args.no_hadron:
        p.readString('HadronLevel:all = off')
    if args.shower_vars:
        p.readString('UncertaintyBands:doVariations = on')
        p.readString('UncertaintyBands:List = {isrHi isr:muRfac=0.5, isrLo isr:muRfac=2.0, fsrHi fsr:muRfac=0.5, fsrLo fsr:muRfac=2.0}')
    if args.kt_hard is not None:
        p.readString(f'BeamRemnants:primordialKThard = {args.kt_hard}')
    p.readString('Beams:setProductionScalesFromLHEF = off')
    p.readString('Print:quiet = on')
    p.init()

    W_scale = None
    if args.rwl:
        print('pre-scanning LHE for <weights> blocks...', flush=True)
        Wl=[]; inw=False; cur=[]
        opener = open
        if args.lhe.endswith('.gz'):
            import gzip; opener = lambda f: gzip.open(f,'rt')
        # two encodings in the wild: LHEF3 compact <weights> blocks (POWHEG
        # rwl) and per-event <rwgt><wgt id=..>v</wgt></rwgt> blocks (MG5)
        mode = None
        with opener(args.lhe) as fh:
            for line in fh:
                t=line.strip()
                if t.startswith('<weights>'): mode='w'; cur=[]; continue
                if t.startswith('</weights>'): mode=None; Wl.append(cur); continue
                if t.startswith('<rwgt>'): mode='r'; cur=[]; continue
                if t.startswith('</rwgt>'): mode=None; Wl.append(cur); continue
                if mode=='w':
                    cur.extend(float(x) for x in t.split())
                elif mode=='r' and t.startswith('<wgt'):
                    cur.append(float(t.split('>',1)[1].split('<',1)[0]))
        W_scale = np.asarray(Wl, dtype=np.float64)
        if W_scale.ndim != 2 or not len(W_scale):
            raise RuntimeError(f'--rwl: no per-event weight blocks found '
                               f'(parsed shape {W_scale.shape})')
        print(f'  {W_scale.shape[0]} events x {W_scale.shape[1]} weights', flush=True)

    out = {k: [] for k in ('l_plus', 'l_minus', 'mll', 'pT_ll', 'y_ll',
                            'dy_pm', 'dphi_pm', 'weight', 'in_fo', 'partons', 'w_scale', 'w_shower',
                            'l_plus_born', 'l_minus_born')}
    save_partons = args.no_hadron  # only meaningful at parton level
    n_total = 0; n_fo = 0; n_wmis = 0; n_wchk = 0
    sum_w_total = 0.0; sum_w_fo = 0.0
    t0 = time.time()

    while True:
        if not p.next():
            if p.infoPython().atEndOfFile():
                break
            continue
        weight = p.infoPython().weight()
        if args.shower_vars:
            info=p.infoPython()
            wrow_sv=[info.weightValueByIndex(i)*weight/max(info.weightValueByIndex(0),1e-300) for i in range(info.numberOfWeights())]
        if W_scale is not None:
            wrow = W_scale[n_total]  # n_total counts previous successes = 0-based index
            # Alignment guard, RATE-based: the first stored weight should track
            # the event weight, but per-event agreement differs by source --
            # MG5's stored central matches XWGTUP to per-mille, while POWHEG's
            # compute_rwgt REPLAY carries per-event reconstruction noise (11%
            # rms, rare large outliers) that cancels in the w_i/w_0 ratios
            # consumers use.  A real OFFSET misalignment decorrelates nearly
            # every event, so we count loose per-event failures and abort only
            # if their rate exceeds 5%.
            if abs(wrow[0]-weight) > 0.5*max(abs(weight),1e-30):
                n_wmis += 1
            n_wchk += 1
            if n_wchk >= 2000 and n_wmis > 0.05*n_wchk:
                raise RuntimeError(f'rwl alignment lost: {n_wmis}/{n_wchk} events '
                                   f'mismatch (last: LHE {wrow[0]} vs pythia {weight})')
        n_total += 1
        sum_w_total += weight
        if args.max_events and n_total > args.max_events:
            n_total -= 1; break

        leps = find_dressed_leptons(p.event)
        if leps is None:
            out['l_plus'].append(np.zeros(4)); out['l_minus'].append(np.zeros(4))
            out['l_plus_born'].append(np.zeros(4)); out['l_minus_born'].append(np.zeros(4))
            out['mll'].append(0.); out['pT_ll'].append(0.); out['y_ll'].append(0.)
            out['dy_pm'].append(0.); out['dphi_pm'].append(0.)
            out['weight'].append(weight); out['in_fo'].append(False)
            if W_scale is not None: out['w_scale'].append(wrow)
            if args.shower_vars: out['w_shower'].append(wrow_sv)
            if save_partons:
                out['partons'].append(np.zeros((0, 4), dtype=np.float32))
            continue
        l_plus, l_minus = leps
        born = find_born_leptons(p.event)
        l_plus_born, l_minus_born = born if born is not None else (np.zeros(4), np.zeros(4))

        # Collect non-lepton final-state particles (partons in --no-hadron mode)
        partons = []
        if save_partons:
            for i in range(1, p.event.size()):
                pp = p.event[i]
                if not pp.isFinal(): continue
                if abs(pp.id()) in (11, 13): continue  # leptons handled separately
                if pp.pT() < 1.0: continue  # IR cutoff for storage
                partons.append((pp.px(), pp.py(), pp.pz(), pp.e()))

        pT_p, y_p, phi_p = to_ptyphi(l_plus)
        pT_m, y_m, phi_m = to_ptyphi(l_minus)
        in_fo_lep = (pT_p > args.pT_lep and pT_m > args.pT_lep and
                     abs(y_p) < args.eta_lep and abs(y_m) < args.eta_lep)

        s = l_plus + l_minus
        px, py, pz, E = s
        mll2 = E*E - px*px - py*py - pz*pz
        mll = float(np.sqrt(max(mll2, 0.0)))
        pT_ll = float(np.sqrt(px*px + py*py))
        y_ll = float(0.5 * np.log((E + pz) / max(E - pz, 1e-30)))
        dy_pm = float(y_p - y_m)
        dphi_pm = float(np.arctan2(np.sin(phi_p - phi_m), np.cos(phi_p - phi_m)))

        in_fo = in_fo_lep and (mll >= args.mll_min) and (mll <= args.mll_max)

        out['l_plus'].append(l_plus); out['l_minus'].append(l_minus)
        out['l_plus_born'].append(l_plus_born); out['l_minus_born'].append(l_minus_born)
        out['mll'].append(mll); out['pT_ll'].append(pT_ll); out['y_ll'].append(y_ll)
        out['dy_pm'].append(dy_pm); out['dphi_pm'].append(dphi_pm)
        out['weight'].append(weight); out['in_fo'].append(in_fo)
        if W_scale is not None: out['w_scale'].append(wrow)
        if args.shower_vars: out['w_shower'].append(wrow_sv)
        if save_partons:
            out['partons'].append(np.asarray(partons, dtype=np.float32) if partons
                                  else np.zeros((0, 4), dtype=np.float32))
        if in_fo:
            n_fo += 1; sum_w_fo += weight

        if n_total % 50000 == 0:
            dt = time.time() - t0
            rate = n_total / max(dt, 1e-6)
            print(f'  total={n_total:,}  fo={n_fo:,}  rate={rate:.0f} ev/s', flush=True)

    arrs = {}
    for k, v in out.items():
        if k == 'partons':
            if save_partons:
                arrs[k] = np.array(v, dtype=object)  # ragged: variable per-event parton count
        else:
            arrs[k] = np.asarray(v)
    np.savez_compressed(
        args.out,
        n_total=np.int64(n_total),
        sum_w_total=np.float64(sum_w_total),
        sum_w_fo=np.float64(sum_w_fo),
        sigma_FO=np.float64(sum_w_fo / max(n_total, 1)),
        **arrs,
    )
    print(f'Saved {args.out}: {n_fo:,} FO / {n_total:,} total. '
          f'σ_FO = {sum_w_fo / max(n_total, 1):.4e} pb', flush=True)
    hard_exit(0)


if __name__ == '__main__':
    main()
