"""runs/parity/report.md + report.csv from every result.json (suite.py report).

Reference per case and tier: the foam reference run (tier A rhoPimpleFoam,
tier B simpleFoam, tier Bc rhoSimpleFoam, finest body-fitted mesh; without a
reference run, the finest ladder level if the case sets ref_fallback_finest,
else MISSING) and, when its ladder converges
monotonically, the Richardson value through its three finest meshes; errors
are measured against the Richardson value when it exists, else the fine run.
A result.json carrying "stale" is listed for the record and takes no part in
references, errors or cost curves. Every row names its U convection scheme.
"""
import csv
import glob
import json
import math
import os

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs", "parity")
NOTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_notes.md")
APP = {"A": "rhoPimpleFoam", "B": "simpleFoam", "Bc": "rhoSimpleFoam"}
DNS_SCHEME = "central (dns)"
D_BOX = math.pi / 4          # dns compressible D (box units), as suite.py


def u_scheme(r):
    """U convection of a row: dns is central; a foam run records its own
    (pre-W5 runs did not, and all of them were linearUpwind)."""
    if r.get("solver") == "dns":
        return DNS_SCHEME
    return r.get("meta", {}).get("u_scheme", "linearUpwind")


def load(path):
    with open(path) as f:
        return json.load(f)


def results(tier, case):
    out = {}
    for f in glob.glob(os.path.join(RUNS, tier, case, "*", "result.json")):
        r = load(f)
        out[os.path.basename(os.path.dirname(f))] = r
    for f in glob.glob(os.path.join(RUNS, "ref", tier, case, "*", "result.json")):
        r = load(f)
        out["ref_" + os.path.basename(os.path.dirname(f))] = r
    return out


def foam_h(kind, lvl):
    """Representative cell size of a foam level (relative)."""
    if kind == "cylinder":
        return 1.0 / lvl[0]
    if kind == "section":
        return 1.0 / lvl[0]
    return 1.0 / (lvl[0] * 2 ** lvl[1])


def richardson(hs, qs):
    """q = q0 + C h^p through three (h, q), h1 > h2 > h3; (q0, p) or None."""
    (h1, q1), (h2, q2), (h3, q3) = sorted(zip(hs, qs), reverse=True)
    d1, d2 = q2 - q1, q3 - q2
    if d1 == 0 or d2 == 0 or d1 * d2 < 0:
        return None
    lo, hi = 0.05, 8.0

    def f(p):
        return (h2 ** p - h1 ** p) / (h3 ** p - h2 ** p) - d1 / d2
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) * flo > 0:
            lo, flo = mid, f(mid)
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    C = d2 / (h3 ** p - h2 ** p)
    return q3 - C * h3 ** p, p


def reference(cfg, name, tier, res):
    c = cfg["cases"][name]
    obs = observables(c)
    ladder = [(k, r) for k, r in res.items() if r.get("solver") == "foam" and "observables" in r
              and r.get("np", cfg["suite"]["np"]) == cfg["suite"]["np"] and not r.get("stale")]
    if not ladder:
        return None
    pts = sorted(((foam_h(c["kind"], r["meta"]["level"]), r) for _, r in ladder),
                 key=lambda x: -x[0])
    ref = [r for k, r in ladder if k.startswith("ref_")]
    if not ref and not c.get("ref_fallback_finest"):
        return None
    fine_src = "reference run" if ref else "finest ladder level (no reference run)"
    ref = ref[0] if ref else pts[-1][1]
    out = dict(fine=ref["observables"], value={}, order={}, source={}, fine_src=fine_src,
               fine_level=ref.get("level"))
    for o in obs:
        out["value"][o] = ref["observables"][o]
        out["source"][o] = "fine"
        if len(pts) >= 3:
            hs = [p[0] for p in pts[-3:]]
            qs = [p[1]["observables"][o] for p in pts[-3:]]
            rr = richardson(hs, qs)
            if rr and 0.5 <= rr[1] <= 4.0:
                out["value"][o], out["order"][o] = rr
                out["source"][o] = "Richardson"
    return out


def observables(c):
    return c["observables"]


def err(c, o, q, ref):
    sc = abs(ref["value"][c.get("scale_by", {}).get(o, o)])
    return abs(q - ref["value"][o]) / sc if sc else float("nan")


def label_res(r):
    x = r.get("resolution", {})
    for k in ("D_h", "c_h", "D_hfine", "nq", "nt", "D_hblade"):
        if k in x:
            v = x[k]
            return f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
    return ""


def cost_to(levels, target):
    """Cheapest cost at which the error stays <= target from then on,
    log-log interpolated between the bracketing levels."""
    lv = sorted([l for l in levels if math.isfinite(l[0]) and math.isfinite(l[1])],
                key=lambda l: l[0])
    if not lv:
        return "n/a"
    k = None
    for i in range(len(lv)):
        if all(e <= target for _, e in lv[i:]):
            k = i
            break
    if k is None:
        return f"> {lv[-1][0]:.0f} s (not reached; best {min(e for _, e in lv) * 100:.2f} %)"
    if k == 0:
        return f"<= {lv[0][0]:.1f} s (coarsest already {lv[0][1] * 100:.2f} %)"
    (c0, e0), (c1, e1) = lv[k - 1], lv[k]
    if e1 <= 0:
        return f"{c1:.1f} s"
    t = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
    return f"{math.exp(math.log(c0) + t * (math.log(c1) - math.log(c0))):.1f} s"


def fmt(v, n=5):
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{n}g}" if math.isfinite(v) else "nan"
    return str(v)


def zone_study(cfg, name, c):
    """Rotor MRF zone sensitivity: calibration runs (`run --zone-r R --tag T`)
    against the same tier/level run at the case's zone_r, refinement fixed."""
    if c["kind"] != "rotor":
        return []
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, "calib", "*", name, "*", "result.json"))):
        r = load(f)
        zr = r.get("meta", {}).get("zone_r")
        if r.get("solver") != "foam" or zr is None or zr == c["zone_r"] or "observables" not in r:
            continue
        bp = os.path.join(RUNS, r["tier"], name, f"foam_{r['level']}", "result.json")
        if not os.path.exists(bp):
            continue
        b = load(bp)
        d = {o: r["observables"][o] / b["observables"][o] - 1.0 for o in c["observables"]}
        out.append(f"| {r['tier']} ({r['app']}) | {r['level']} | {u_scheme(r)} | {zr:g} | "
                   f"{r['meta'].get('refine_r', zr):g} | {r['cells']} | "
                   + " | ".join(f"{fmt(r['observables'][o], 6)} ({d[o] * 100:+.2f}%)"
                                for o in c["observables"]) + " |")
    if not out:
        return []
    return ([f"\n## {name}: MRF zone sensitivity\n",
             f"Sharp foam zone radius moved across dns' blend band (f = 1 to f = 0); the "
             f"refinement cylinders stay at the case's zone_r = {c['zone_r']:g} D. Delta = "
             f"q / q(zone_r {c['zone_r']:g}) - 1, same tier and level (+ = larger magnitude).\n",
             "| tier | level | U scheme | zone_r / D | refine_r / D | cells | "
             + " | ".join(c["observables"]) + " |",
             "|---" * (6 + len(c["observables"])) + "|"] + out + [""])


def dns_ladder(name, c, tier, res, ref):
    """Rotor parity verdict (cases.toml `parity_bands`): the dns ladder in the
    tier dns is scored in, its monotonicity and Richardson value (h = h_fine,
    the largest axis), and PARITY iff the verdict value -- Richardson when the
    ladder is monotone and the fit exists, else the finest level -- is within
    the band of the reference for every observable. Bands are fixed before
    measuring and never widened after."""
    bands = c.get("parity_bands")
    if not bands or c["kind"] != "rotor" or tier != c.get("dns_scored_in", tier) or not ref:
        return []
    rows = sorted(((max(r["resolution"]["hfine_axes"].values()), k, r) for k, r in res.items()
                   if r.get("solver") == "dns" and "observables" in r and not r.get("stale")
                   and not k.startswith("ref_")), key=lambda x: -x[0])
    if not rows:
        return []
    obs = observables(c)
    md = [f"\n## {name}: dns ladder and parity verdict (tier {tier}, {APP[tier]})\n",
          "Bands: " + ", ".join(f"{o} {bands[o] * 100:g} %" for o in obs)
          + f"; reference = {APP[tier]} " + ", ".join(
              f"{o} {fmt(ref['value'][o], 6)} ({ref['source'][o]})" for o in obs)
          + ". BL = D/sqrt(Re)/h_fine,min cells.\n",
          "| level | D/h_fine | BL cells | " + " | ".join(f"{o} | d{o} %" for o in obs) + " |",
          "|---" * (3 + 2 * len(obs)) + "|"]
    for h, k, r in rows:
        bl = D_BOX / math.sqrt(c["re"]) / min(r["resolution"]["hfine_axes"].values())
        md.append(f"| {k} | {D_BOX / h:.1f} | {bl:.2f} | " + " | ".join(
            f"{fmt(r['observables'][o], 6)} | {(r['observables'][o] / ref['value'][o] - 1) * 100:+.2f}"
            for o in obs) + " |")
    verdict, parts = True, []
    for o in obs:
        hs = [x[0] for x in rows[-3:]]
        qs = [x[2]["observables"][o] for x in rows[-3:]]
        mono = len(qs) >= 3 and (qs[1] - qs[0]) * (qs[2] - qs[1]) > 0
        rr = richardson(hs, qs) if mono else None
        q, src = (rr[0], f"Richardson p = {rr[1]:.2f}") if rr else (qs[-1], "finest level")
        d = q / ref["value"][o] - 1
        ok = abs(d) <= bands[o]
        verdict &= ok
        parts.append(f"{o}: ladder {'monotone' if mono else 'NOT monotone'}, {src} "
                     f"{fmt(q, 6)} -> {d * 100:+.2f} % ({'within' if ok else 'OUTSIDE'} "
                     f"{bands[o] * 100:g} %)")
    md += [""] + [f"- {p}" for p in parts] + [
        f"- **verdict: {'PARITY' if verdict else 'NOT AT PARITY'}**", ""]
    return md


def split_table(name, c):
    """Rotor CT/CQ split (tools/parity/split_dns.py --json into
    runs/parity/split/<case>/<label>.json): hub vs blade x pressure /
    convective / viscous per source, and each part's share of the dns - foam
    gap against the scored reference (`dns_scored_in` L3)."""
    files = sorted(glob.glob(os.path.join(RUNS, "split", name, "*.json")))
    if c["kind"] != "rotor" or not files:
        return []
    sp = {os.path.splitext(os.path.basename(f))[0]: load(f) for f in files}
    md = [f"\n## {name}: CT/CQ split, hub vs blade (tools/parity/split_dns.py)\n",
          "dns: the ghost-wall tally per interface face (`--emit surface`), final state "
          "at 4.5 D/U; foam: per-face forces of the converged L3. Hub = r <= r_hub + T and "
          "|x| <= hub half-length + T, ONE T for every source of a case: the coarsest dns "
          "level's sqrt(median face area), the staircase's offset from the true surface. "
          "p / c / v = pressure / convective (dns only: the staircase mirror's mass flux "
          "through the moving wall) / viscous. On a staircase the p/v division of a sloped "
          "surface's stress is not the body-fitted one (they converge only with D/h); the "
          "per-part TOTALS are comparable.\n",
          "| source | part | CT | CT_p | CT_c | CT_v | CQ | CQ_p | CQ_c | CQ_v |",
          "|---" * 10 + "|"]
    for k, t in sp.items():
        for part in ("total", "hub", "blade"):
            q = t[part]
            md.append(f"| {k} (T {t['tol']:.4f}) | {part} | " + " | ".join(
                f"{q[o]:+.5f}" for o in ("CT", "CT_p", "CT_c", "CT_v", "CQ", "CQ_p",
                                         "CQ_c", "CQ_v")) + " |")
    ref = sp.get(f"foam_{c.get('dns_scored_in', 'B')}_L3")
    if ref:
        md += ["", f"Gap against foam_{c.get('dns_scored_in', 'B')}_L3 (dns - foam; "
               "share = part / total gap):", ""]
        for k, t in sp.items():
            if not k.startswith("dns"):
                continue
            g = t["total"]["CT"] - ref["total"]["CT"]
            parts = []
            for part in ("hub", "blade"):
                d = t[part]["CT"] - ref[part]["CT"]
                parts.append(f"{part} {d:+.5f} ({d / g * 100:+.0f} %)")
            parts.append("blade p {:+.5f}, c {:+.5f}, v {:+.5f}".format(
                *(t["blade"][o] - ref["blade"][o] for o in ("CT_p", "CT_c", "CT_v"))))
            q, qr = t["total"], ref["total"]
            md.append(f"- {k}: CT gap {g:+.5f} ({(q['CT'] / qr['CT'] - 1) * 100:+.1f} %): "
                      + "; ".join(parts) + f". CQ {(q['CQ'] / qr['CQ'] - 1) * 100:+.2f} %, "
                      f"of which convective {q['CQ_c'] / qr['CQ'] * 100:+.2f} % "
                      f"(CQ - CQ_c: {((q['CQ'] - q['CQ_c']) / qr['CQ'] - 1) * 100:+.2f} %)")
    return md + [""]


def main(cfg):
    rows, md = [], []
    prov = None
    revs = {}  # git_rev -> recorded dates of the non-stale rows that carry it
    budget_routine = 0.0
    md.append("# dns vs OpenFOAM parity suite\n")
    md.append("Generated by `tools/parity/suite.py report` from `runs/parity/*/*/*/result.json`. "
              "Spec: `tools/parity/DESIGN.md`; case definitions: `tools/parity/cases.toml`.\n")
    summary = []
    for name, c in cfg["cases"].items():
        for tier in c["tiers"]:
            res = results(tier, name)
            if tier in ("B", "Bc") and c.get("b_dns") == "reuse_A":
                for k, r in results("A", name).items():
                    if r.get("solver") == "dns" and not k.startswith("ref_") and not r.get("stale"):
                        res[k] = dict(r, reused_from="A")
            if not res:
                continue
            ref = reference(cfg, name, tier, res)
            obs = observables(c)
            app = APP[tier]
            # re-scored dns rows in a tier other than dns_scored_in (rotor: tier B, the
            # M -> 0 leg; dns is compressible and scored against Bc) are informational
            dns_info = tier != "A" and c.get("dns_scored_in", tier) != tier
            md.append(f"\n## {name}, tier {tier} ({app})\n")
            if c.get("u_scheme_note"):
                md.append(f"**foam U: {c['u_scheme_note']}.**\n")
            if dns_info:
                md.append(f"dns rows here are for information only (this tier is the M -> 0 leg): "
                          f"dns is scored in tier {c['dns_scored_in']} ({APP[c['dns_scored_in']]}).\n")
            if ref:
                md.append("Reference (" + app + ", " + ref["fine_src"] + "): " + ", ".join(
                    f"{o} = {fmt(ref['value'][o], 6)} ({ref['source'][o]}"
                    + (f", p = {ref['order'][o]:.2f}" if o in ref["order"] else "")
                    + f"; fine {fmt(ref['fine'][o], 6)})" for o in obs) + "\n")
                es = {r.get("meta", {}).get("e_scheme") for r in res.values()
                      if r.get("solver") == "foam" and not r.get("stale")} - {None}
                if es:
                    md.append("Energy convection (foam): " + ", ".join(f"`{e}`" for e in sorted(es))
                              + " (the parity rule covers U only).\n")
            else:
                md.append("Reference: MISSING\n")
            md.append("| solver | level | U scheme | resolution | cells | " + " | ".join(obs)
                      + " | err max % | t_settle | setup s | solve s | total s | run s | GPU J (settle) |")
            md.append("|---" * (12 + len(obs)) + "|")
            curves = {"dns": [], "foam": []}
            for key in sorted(res, key=lambda k: (bool(res[k].get("stale")),
                                                  res[k].get("solver", ""), k)):
                r = res[key]
                sch = u_scheme(r)
                if "observables" not in r:
                    md.append(f"| {r.get('solver')} | {key} | {sch} | {label_res(r)} | "
                              f"{r.get('cells')} | " + " | ".join("-" for _ in obs)
                              + f" | FAILED: {r.get('error')} |" + " |" * 6)
                    continue
                o_ = r["observables"]
                stale = r.get("stale")
                errs = {o: err(c, o, o_[o], ref) for o in obs} if ref and not stale else {}
                emax = max(errs.values()) if errs else float("nan")
                np_ok = r.get("np", cfg["suite"]["np"]) == cfg["suite"]["np"]
                info = stale or (dns_info and r["solver"] == "dns")
                if not key.startswith("ref_") and np_ok and not info:
                    curves[r["solver"]].append((r["total_s"], emax))
                    if r.get("reused_from") is None:
                        budget_routine += r["run_wall_s"] + (r["setup_s"] if r["solver"] == "foam" else 0)
                if not stale and r.get("provenance"):
                    p = r["provenance"]
                    prov = prov or p
                    revs.setdefault(p.get("git_rev", "?"), []).append(
                        str(p.get("recorded", "?"))[:10])
                ge = r.get("gpu_energy_settle_J")
                lab = key + (f" (np {r['np']})" if r.get("np") and not np_ok else "") \
                    + (" [tier A run]" if r.get("reused_from") else "") \
                    + (" [adopted]" if r.get("adopted") else "") \
                    + (" [M->0 leg]" if dns_info and r["solver"] == "dns" else "") \
                    + (f" **STALE: {stale}**" if stale else "")
                md.append(f"| {r['solver']} | {lab} | {sch} | {label_res(r)} | {r['cells']} | "
                          + " | ".join(f"{fmt(o_[o], 6)} ({errs.get(o, float('nan')) * 100:+.2f}%)"
                                       if errs else fmt(o_[o], 6) for o in obs)
                          + f" | {emax * 100:.2f} | {fmt(r['t_settle'], 4)} | {r['setup_s']:.1f} | "
                          f"{r['solve_s']:.1f} | {r['total_s']:.1f} | {r['run_wall_s']:.1f} | "
                          f"{fmt(ge, 4) if isinstance(ge, float) else 'n/a'} |")
                rows.append(dict(case=name, tier=tier, solver=r["solver"],
                                 app=r.get("app", "dns"), level=key, np=r.get("np", ""),
                                 u_scheme=sch, stale=stale or "",
                                 resolution=label_res(r), cells=r["cells"],
                                 **{f"q_{o}": o_[o] for o in obs},
                                 **{f"err_{o}": errs.get(o, "") for o in obs},
                                 err_max=emax, t_settle=r["t_settle"], t_end=r.get("t_end"),
                                 setup_s=r["setup_s"], solve_s=r["solve_s"],
                                 total_s=r["total_s"], run_wall_s=r["run_wall_s"],
                                 gpu_energy_settle_J=ge if isinstance(ge, float) else "",
                                 gpu_energy_total_J=r.get("gpu_energy_total_J", ""),
                                 cpu_energy_J="n/a", steps=r.get("steps", ""),
                                 reused_from=r.get("reused_from", "")))
            md.append("")
            for s in ("dns", "foam"):
                if s == "dns" and dns_info:
                    continue
                c1, c3 = cost_to(curves[s], 0.01), cost_to(curves[s], 0.03)
                if s == "foam" and ref and ref["fine_src"] != "reference run":
                    # the finest level scores itself 0: the curve is self-convergence
                    c1, c3 = (x + " (vs finest level, no ref run)" for x in (c1, c3))
                md.append(f"- cost to 1 % ({s}): {c1}; to 3 %: {c3}")
                summary.append((name, tier, s, c1, c3))
            md += dns_ladder(name, c, tier, res, ref)
        md += zone_study(cfg, name, c)
        md += split_table(name, c)
        if c.get("notes"):
            md += [f"\n## {name}: notes\n"] + [f"- {n}" for n in c["notes"]] + [""]
    md.insert(2, "\n## Verdict (cost = wall time from process start to t_settle, incl. setup)\n\n"
              "| case | tier | solver | cost to 1 % | cost to 3 % |\n|---|---|---|---|---|\n"
              + "\n".join(f"| {a} | {b} | {s} | {x} | {y} |" for a, b, s, x, y in summary) + "\n")
    if prov:
        md.insert(3, "\n## Provenance\n\n"
                  "Git revisions across the non-stale rows (each row's own provenance is in its "
                  "`result.json`):\n\n"
                  + "\n".join(f"- {rv}: {len(ds)} rows, recorded {min(ds)}"
                              + (f" .. {max(ds)}" if max(ds) != min(ds) else "")
                              for rv, ds in sorted(revs.items(), key=lambda x: min(x[1])))
                  + "\n\nFirst non-stale row's provenance (host and toolchain):\n\n"
                  + "\n".join(f"- {k}: {v}" for k, v in prov.items())
                  + f"\n- foam ranks: {cfg['suite']['np']} (P-cores, `--bind-to core --map-by core`)"
                  f"; scaling point {cfg['suite']['np_scaling']} ranks (6 P + 6 E cores)\n"
                  f"- CPU energy: n/a (RAPL `energy_uj` is root-only); GPU energy: `rocm-smi "
                  f"--showpower` sampled at 1 Hz, trapezoid-integrated over the dns process\n"
                  f"- routine wall time spent (all non-reference runs incl. foam setup): "
                  f"{budget_routine / 60:.1f} min (budget {cfg['suite']['suite_cap_s'] / 60:.0f} min)\n")
    if os.path.exists(NOTES):
        md.append("\n" + open(NOTES).read())
    with open(os.path.join(RUNS, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(RUNS, "report.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: fmt(v, 8) if isinstance(v, float) else v for k, v in r.items()})
    print(f"wrote {os.path.join(RUNS, 'report.md')} and report.csv ({len(rows)} rows)")
