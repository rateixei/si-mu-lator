![Si-μ-lator](logo/black@3x.png?raw=true "Title")

## Si-μ-lator: Package for toy muon detector simulation

## Usage

```
usage: si_mu_late.py [-h] -d DETCARD -n NEVS [-m] [-x muxmin muxmax] [-y muymin muymax] [-a muamin muamax] [-b BKGR]

optional arguments:
  -h, --help            show this help message and exit
  -d DETCARD, --detector DETCARD
                        Detector card
  -n NEVS, --nevents NEVS
                        Number of events
  -m, --addmuon         Simulate muon
  -x muxmin muxmax, --muonx muxmin muxmax
                        Generated muon window in X (leave empty for 0)
  -y muymin muymax, --muony muymin muymax
                        Generated muon window in Y (leave empty for 0)
  -a muamin muamax, --muona muamin muamax
                        Generated muon angle window (leave empty for 0)
  -b BKGR, --bkgrate BKGR
                        Background rate (Hz) per plane
```

## Setup environment at SLAC
```
source init_sing.sh
```
