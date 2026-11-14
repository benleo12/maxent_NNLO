"""Declared W prediction columns and correlated acceptance/asymmetry transforms.

These functions only construct observables. They neither select constraints nor
read reference values. Call them before freezing a fit, then evaluate the saved
columns against the independent integration after the training choice is fixed.
"""
import numpy as np

from maxent_upgrade import stack_predictions
if __package__:
    from .w_coordinates import coordinates
else:
    from w_coordinates import coordinates


ETA_EDGES = np.array([0., .21, .42, .63, .84, 1.05,
                     1.37, 1.52, 1.74, 1.95, 2.18, 2.5])
CUTS = ("pt28", "pt32", "hard", "pt32hard", "mt60", "combined")


def observable_groups(events, *, exploratory=False):
    """Return named columns, labels and exact definitions for the fixed W tests.

    Inputs already satisfy the original baseline. All sums keep that denominator.
    Lepton rapidity is used, as in the massless fixed-order booking; the retained
    shower four-vectors have a tiny mass-dependent difference from pseudorapidity.
    Optional old-cut ledgers are explicitly exploratory, never new validation.
    """
    e = coordinates(events["l_born"], events["nu"])
    qt, pt, y = e["pt_w"], e["pt_l"], np.abs(e["y_l"])
    masks = dict(pt28=pt > 28, pt32=pt > 32, hard=qt > 50,
                 pt32hard=(pt > 32) & (qt > 50), mt60=e["mt"] > 60,
                 combined=(e["mt"] > 60) & (e["pt_nu"] > 30) & (y < 1.7))
    definitions = dict(
        pt28="charged-lepton pT>28 GeV", pt32="charged-lepton pT>32 GeV",
        hard="W qT>50 GeV", pt32hard="charged-lepton pT>32 GeV and W qT>50 GeV",
        mt60="lepton-neutrino transverse mass>60 GeV",
        combined="transverse mass>60 GeV, neutrino pT>30 GeV and abs(lepton rapidity)<1.7")
    groups = dict(cuts=dict(
        values=np.column_stack([np.ones(len(pt))] + [masks[k] for k in CUTS]),
        labels=("norm_born",) + tuple("hold_" + k for k in CUTS),
        definition=dict(role="validation", observable="baseline and selected absolute rates",
                        selections=definitions, denominator="original baseline", units="pb")))
    for key, cut in (("eta28", "pt28"), ("eta32", "pt32"), ("eta_mt60", "mt60")):
        columns = [(y >= lo) & (y < hi) & masks[cut]
                   for lo, hi in zip(ETA_EDGES[:-1], ETA_EDGES[1:])]
        groups[key] = dict(values=np.column_stack(columns).astype(float),
                           labels=tuple(
                               f"{key}_{i:02d}" for i in range(len(columns))),
                           definition=dict(role="validation", observable="absolute charged-lepton rapidity",
                                           selection=definitions[cut], edges=ETA_EDGES.tolist(
                                           ),
                                           units="pb integrated per bin", density_label=r"$d\sigma/d|y_\ell|$ [pb]",
                                           convention="charged pre-QED lepton; massless fixed-order eta equals rapidity"))
    if exploratory:
        groups["old_cut_ledger"] = _ledger(e)
    return groups


def _ledger(e):
    qt, c, mass, pt = e["pt_w"], e["c"], e["m_w"], e["pt_l"]
    iq = np.where(qt >= 50, 2, np.where(qt >= 25, 1, 0))
    ia = np.where(abs(c) >= .7, 2, np.where(abs(c) > .6, 1, 0))
    im = np.where(mass >= 84, 2, np.where(mass >= 76, 1, 0))
    born_pt = .5 * mass * np.sqrt(np.maximum(1 - c * c, 0))
    columns, labels = [], []
    for category, selection in (("qc", "pt0"), ("qc", "pt27"), ("qc", "pt30"),
                                ("qc", "proj27"), ("qc", "proj30"),
                                ("qm", "pt0"), ("qm", "pt27"), ("qm", "pt30")):
        accepted = (born_pt > int(selection[4:]) if selection.startswith("proj")
                    else pt > int(selection[2:]))
        for recoil in range(3):
            for region in range(3):
                columns.append((iq == recoil) & (
                    (ia if category == "qc" else im) == region) & accepted)
                labels.append(f"expl_{category}_{recoil}_{region}_{selection}")
    return dict(values=np.column_stack(columns).astype(float), labels=tuple(labels),
                definition=dict(role="exploratory", observable="old-cut disjoint-region ledger",
                                recoil_edges_GeV=[0, 25, 50, None], polar_edges=[0, .6, .7, 1],
                                mass_edges_GeV=[0, 76, 84, None], lepton_cuts_GeV=[27, 30],
                                projection="b=(m/2)*sqrt(1-c^2); I=I0+(I-I0)",
                                denominator="original baseline", units="pb"))


def acceptances(prediction):
    """Normalize raw cut rates, retaining common numerator/denominator covariance."""
    labels = ("norm_born",) + tuple("hold_" + k for k in CUTS)
    if prediction.labels != labels:
        raise ValueError(
            "acceptances require the declared raw cuts group in its original order")
    numerator, denominator = np.r_[np.arange(1, 7), 4], np.r_[
        np.zeros(6, dtype=int), 3]
    if (prediction.mean[:, denominator] <= 0).any():
        raise ValueError(
            "all acceptance denominators must be positive at every coherent scale")
    central = prediction.mean[0]
    jac = np.zeros((7, 7))
    jac[np.arange(7), numerator] = 1 / central[denominator]
    jac[np.arange(7), denominator] -= central[numerator] / \
        central[denominator]**2
    return prediction.transform(lambda a: a[:, numerator] / a[:, denominator], jac,
                                labels=CUTS + ("pt32_given_hard",),
                                definition="fractions of the baseline; final entry conditioned on W qT>50 GeV")


def charge_asymmetry(plus, minus):
    """Form (W+−W−)/(W++W−) from absolute spectra with their paired covariance."""
    if plus.labels != minus.labels:
        raise ValueError("charge spectra must use identical observable labels")
    n = len(plus.labels)
    joint = stack_predictions([plus, minus])
    total = joint.mean[:, :n] + joint.mean[:, n:]
    if (total <= 0).any():
        raise ValueError(
            "the charge sum must be positive in every bin and coherent scale")
    p, m = plus.mean[0], minus.mean[0]
    jac = np.concatenate([np.diag(2 * m / (p + m)**2),
                         np.diag(-2 * p / (p + m)**2)], axis=1)
    return joint.transform(lambda a: (a[:, :n] - a[:, n:]) / (a[:, :n] + a[:, n:]), jac,
                           labels=plus.labels, definition="charge asymmetry from absolute W+ and W− rates")
