#!/usr/bin/env python3
"""Independent transcription of the Charlette, Meneveau & Veynante (2002)
efficiency function (static beta, the saturated "Delta/delta_L - 1" form of
Wang, Boileau & Veynante 2011) that `dns flame --efficiency charlette` uses
(src/dns/tfles.hpp, charletteEfficiency). Writes the reference table the host
test `tfles` compares the C++ against (1e-12 relative):

    tfles_eff.py > tests/data/chem/tfles_charlette.csv

Columns: r = Delta/delta_L, u = u'_Delta/S_L, re = u'_Delta Delta/nu,
beta, xi. Pure Python floats (IEEE double), no numpy.

    Xi    = (1 + min(r - 1, Gamma u))**beta
    Gamma = (((fu**-a + fd**-a)**(-1/a))**-b + fre**-b)**(-1/b)
    fu    = 4 (27 Ck/110)**0.5 (18 Ck/55) u**2
    fd    = (27/110 Ck pi**(4/3) (r**(4/3) - 1))**0.5
    fre   = (9/55 exp(-3/2 Ck pi**(4/3) / re))**0.5 re**0.5
    a     = 0.60 + 0.20 exp(-0.1 u) - 0.20 exp(-0.01 r),  b = 1.4, Ck = 1.5
"""
import math

CK = 1.5
B = 1.4


def xi(r, u, re, beta):
    if not (r > 1 and u > 0 and re > 0):
        return 1.0
    p43 = math.pi ** (4.0 / 3.0)
    fu = 4.0 * math.sqrt(27.0 * CK / 110.0) * (18.0 * CK / 55.0) * u * u
    fd = math.sqrt(27.0 / 110.0 * CK * p43 * (r ** (4.0 / 3.0) - 1.0))
    fre = math.sqrt(9.0 / 55.0 * math.exp(-1.5 * CK * p43 / re)) * math.sqrt(re)
    a = 0.60 + 0.20 * math.exp(-0.1 * u) - 0.20 * math.exp(-0.01 * r)
    fud = (fu ** -a + fd ** -a) ** (-1.0 / a)
    # fre underflows to 0 at tiny Re: IEEE pow(0, -b) = inf, so Gamma = 0
    # (the C++ reaches it through inf; Python raises instead).
    gam = 0.0 if fre == 0.0 else (fud ** -B + fre ** -B) ** (-1.0 / B)
    return (1.0 + min(r - 1.0, gam * u)) ** beta


def main():
    print("r,u,re,beta,xi")
    # 50 points: a deterministic lattice over the TFLES range (F 1-10, u'
    # 0.05-20 S_L, Re_Delta 0.5-500, beta 0.5 and 1) plus the edge cases.
    pts = []
    rs = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    us = [0.0, 0.05, 0.5, 2.0, 8.0, 20.0]
    k = 0
    for r in rs:
        for u in us:
            re = 0.5 * 10 ** (k % 4) * (1 + 0.37 * (k % 3))
            beta = 0.5 if k % 5 else 1.0
            pts.append((r, u, re, beta))
            k += 1
    pts += [(4.0, 3.0, 1e-3, 0.5), (4.0, 3.0, 1e4, 0.5), (1.0001, 1.0, 10.0, 0.5),
            (2.5, 1e-4, 50.0, 0.5), (7.5, 4.0, 40.0, 0.5), (3.0, 1.0, 25.0, 0.5),
            (6.0, 2.5, 33.3, 0.5), (1.25, 10.0, 200.0, 0.5),
            (9.0, 0.3, 4.0, 0.5), (2.0, 0.8, 7.0, 1.0),
            (3.3, 5.5, 120.0, 0.5), (8.0, 12.0, 300.0, 0.5),
            (4.0, 0.01, 1.0, 0.5), (5.0, 50.0, 1000.0, 0.5)]
    for r, u, re, beta in pts[:50]:
        print(f"{r!r},{u!r},{re!r},{beta!r},{xi(r, u, re, beta)!r}")


if __name__ == "__main__":
    main()
