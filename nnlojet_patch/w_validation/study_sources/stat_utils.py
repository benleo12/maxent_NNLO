"""Coherent channel-wise resampling of independent equal-budget integrations."""
import numpy as np

NBOOT=4000
SEEDS={('T','WP'):133001,('T','WM'):133002,('V','WP'):134001,('V','WM'):134002}

def bootstrap_raw(acc,group,charge,nboot=NBOOT):
    rng=np.random.default_rng(SEEDS[group,charge])
    a=np.asarray(acc)[:,:,:,0]
    out=np.zeros((nboot,a.shape[2]))
    for channel in a:
        valid=np.isfinite(channel).all(axis=1)
        assert np.all(valid | np.isnan(channel).all(axis=1)), 'Partially nonfinite physical accumulator'
        channel=channel[valid]
        assert len(channel)>1
        count=rng.multinomial(len(channel),np.ones(len(channel))/len(channel),size=nboot)
        out+=count@channel/len(channel)
    return out
