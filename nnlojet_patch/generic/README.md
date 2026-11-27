# Run-time moment measurements for NNLOJET 1.0.2

This patch lets the same executable integrate Chebyshev moments of the scalar observables registered for any NNLOJET process. NNLOJET evaluates each measurement in its ordinary histogram loop, including its local subtraction terms, global and histogram-level cuts, and scale variations.

## Install once

Apply the patch to an NNLOJET v1.0.2 source tree. From that tree run

```sh
patch -p1 < /path/to/maxent_NNLO/nnlojet_patch/generic/Reweight.f90.patch
cmake --build build --target NNLOJET --parallel
cmake --install build
```

Use a tree configured with the desired process libraries and install prefix. The modified routine is linked into the core library. Run the executable with the matching rebuilt core and the process libraries from that installation. Adding a moment or switching between registered observables then requires only a new run card and sidecar file. A new partonic process or a new physical observable outside the native NNLOJET catalogue still needs its NNLOJET implementation.

## Integrate the requested measurements

A booking job places `maxent_measures.dat` in NNLOJET's working directory. A three-row example uses the absolute rapidity of a dilepton pair and its transverse momentum.

```text
3
1
abs_yz 1 1 0 2.4 0 0 0 0 0 0
1
ptz 0 2 30 500 0 0 30 60 500 0
2
abs_yz 2 1 0 2.4 0 0 0 0 0 0
ptz 1 2 30 500 0 0 30 60 500 0
```

The first number counts measurement rows. Each row starts with its number of axes (one or two). Each axis occupies one line in this order:

```text
NNLOJET_observable_name degree map_code lo hi mass width a b c d [undefined_value]
```

The hook resolves the observable **name** in the current process at startup. Native NNLOJET IDs depend on which names have already been registered. The map codes are 1 for linear, 2 for logarithmic, and 3 for the relativistic Breit–Wigner mass map. Give the resonance mass and width for code 3 and zero for those fields otherwise. The mapping interval is in the units of the specified observable. Values outside the interval are clipped.

The optional twelfth field supplies a finite scalar only when NNLOJET reports
that the observable is undefined for that configuration. Omitting it retains
the error on an undefined observable. For example,
`ptj2 1 2 30 1000 0 0 30 60 1000 0 0` explicitly assigns zero transverse
momentum when no second jet exists. Its window vanishes at zero, so such events
contribute nothing to this moment but remain in the fiducial cross section.
Use the same convention in the shower observable. This is part of the physical
measurement definition; the code never supplies an implicit zero or chooses a
convention for you. An observable reported as valid but returning a nonfinite
value still stops the run, even when an undefined-value convention is present.
Descriptors contain exactly 11 or 12 whitespace-separated fields.

The four window parameters are zero for an unwindowed axis. Otherwise, the weight rises smoothly in log observable from zero at `a` to one at `b`. Equal positive edges give a lower step inclusive at `a`. With `c=d=0` the window extends upward. Positive `c` and zero `d` give a hard upper cutoff excluding the endpoint `c`. Positive `d>c` gives a smooth fall to zero. The polynomial has degree zero for the window rate. A two-axis row multiplies both axes' windowed polynomials.

The corresponding entries in the run card's ordinary histogram block are

```text
HISTOGRAMS
  cross > fid
  cross > mx0001 fac=mxmoment1
  cross > mx0002 fac=mxmoment2
  cross > mx0003 fac=mxmoment3
END_HISTOGRAMS
```

The filenames after `>` are histogram labels. The positive number after `mxmoment` selects the sidecar row. Here `mx0001` is the raw rapidity-moment integral, `mx0002` the raw recoil-window integral, and `mx0003` their indicated product integral. Each output is a native weighted cross section in fb with one value and integration error per scale (`#nx: 0`). A normalized moment or a cross section in pb is formed from these raw outputs by the booking workflow.

An unknown observable at startup, or one undefined without an explicit convention,
stops the run and identifies the row and observable. A leading-jet transverse
momentum can instead be made well-defined by requiring a jet in the fiducial
selection. For a subleading jet in an inclusive one-jet sample, an explicit zero
convention with a window excluding zero preserves the original fiducial region.
