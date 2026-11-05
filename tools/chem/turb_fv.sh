#!/bin/bash
# M5 physics numbers (docs/fvcomp.md "Turbulence", docs/chem.md "Low-Mach turbulence"):
# the turbulent consumption speed S_c (stats column SL) and the flame-surface ratio
# (area) of a premixed flame box at u'/S_L = 4, l_t = 2 delta_th, structured vs FV, on
# the SAME uniform mesh, every run a simjob <= 295 s. Prints one row per run: mean S_c
# and area over t in [T0, T1].
#   usage: tools/chem/turb_fv.sh DNS OUTDIR [set ...]
#   sets:  comp (C0 DNS, compressible, 3 grids), lm (C0 DNS, low-Mach, 3 grids), lam (laminar S_L),
#          pairs (C0 + LES smag / sigma, TFLES dynamic, Charlette; C2b), lmpairs
set -u
DNS=${1:?dns binary}
OUT=${2:?output dir}
shift 2
SETS=${*:-comp lm pairs lmpairs}
SIMJOB=${SIMJOB:-/home/user/Documents/pesto/build/simjob}
T0=${T0:-0.5}
T1=${T1:-2}
TT=${TT:-2}
mkdir -p "$OUT"

# u'/S_L 4, l_t = 2 delta_th (Re_t 18): the IC box sits on the flame at t = 0, the
# inflow's needs ~13 flame times to arrive, so the window [T0, T1] is the IC's decay.
# grid cpd: --cells-per-delta cpd, ny = nz = 12 cpd (L = 6 l_t), 6 fresh + 4 burnt delta_th.
box() { local cpd=$1; echo "--turb 4 --turb-length 2 --cells-per-delta $cpd --ny $((12 * cpd)) --nz $((12 * cpd)) --upstream 6 --downstream 4 --tt $TT --emit stats --quiet"; }

mean() { # file column: mean over T0 <= t <= T1
    python3 - "$1" "$2" "$T0" "$T1" <<'EOF'
import sys, csv
f, col, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rows = [r for r in csv.reader(l for l in open(f) if not l.startswith('#'))]
h = rows[0]
if col not in h:
    print("n/a"); sys.exit()
i, it = h.index(col), h.index('t')
v = [float(r[i]) for r in rows[1:] if t0 <= float(r[it]) <= t1]
print("%.5f" % (sum(v) / len(v)) if v else "n/a")
EOF
}

run() { # name args...
    local name=$1
    shift
    "$SIMJOB" submit --gpu -t 295 -m 12000 --wait -- "$DNS" flame "$@" --out "$OUT/$name" > "$OUT/$name.job" 2>&1
    local st="$OUT/${name}_stats.csv"
    if [ ! -f "$st" ]; then
        printf "%-28s FAILED (%s)\n" "$name" "$(tail -1 "$OUT/$name.job")"
        return
    fi
    printf "%-28s S_c %s  area %s\n" "$name" "$(mean "$st" SL)" "$(mean "$st" area)"
}

for set in $SETS; do
    case $set in
    comp)
        for cpd in 4 6 8; do
            run "comp_st_$cpd" $(box $cpd) --mach 0.02
            run "comp_fv_$cpd" $(box $cpd) --mach 0.02 --solver fv
        done ;;
    compc) # the FV route compensated (plain FP32 drops sub-ulp heat increments: the FV dt is
           # 2x the structured one's smaller)
        for cpd in 4 6 8; do
            run "comp_fvc_$cpd" $(box $cpd) --mach 0.02 --solver fv --compensated
        done ;;
    lam) # the laminar S_L of each solver on the same h (a thin 3D box): the scale the
         # turbulent S_c gap is read against
        for cpd in 4 6 8; do
            l="--cells-per-delta $cpd --ny 8 --nz 8 --upstream 6 --downstream 4 --tt $TT --emit stats --quiet"
            run "lam_comp_st_$cpd" $l --mach 0.02
            run "lam_comp_fv_$cpd" $l --mach 0.02 --solver fv
            run "lam_lm_st_$cpd" $l --physics lowmach
            run "lam_lm_fv_$cpd" $l --physics lowmach --solver fv
        done ;;
    lm)
        for cpd in 4 6 8; do
            run "lm_st_$cpd" $(box $cpd) --physics lowmach
            run "lm_fv_$cpd" $(box $cpd) --physics lowmach --solver fv
        done ;;
    pairs)
        for m in "les_smag:--les 0.17 --sgs smag" "les_sigma:--les 1.35 --sgs sigma" \
                 "tfles:--tfles 2" "charlette:--tfles 2 --les 1.35 --sgs sigma --efficiency charlette"; do
            n=${m%%:*}
            a=${m#*:}
            run "comp_st_${n}" $(box 4) --mach 0.02 $a
            run "comp_fv_${n}" $(box 4) --mach 0.02 --solver fv $a
        done
        # C2b (rich phi 1.3, the A(phi) law): the structured compressible solver has no
        # two-reactant --turb; its cross-solver partner is the FV low-Mach route.
        run "comp_fv_c2b" $(box 4) --mach 0.02 --solver fv --equivalence 1.3
        run "lm_fv_c2b" $(box 4) --physics lowmach --solver fv --equivalence 1.3
        run "lm_st_c2b" $(box 4) --physics lowmach --equivalence 1.3 ;;
    lmpairs)
        for m in "les_smag:--les 0.17 --sgs smag" "les_sigma:--les 1.35 --sgs sigma" \
                 "tfles:--tfles 2" "charlette:--tfles 2 --les 1.35 --sgs sigma --efficiency charlette"; do
            n=${m%%:*}
            a=${m#*:}
            run "lm_st_${n}" $(box 4) --physics lowmach $a
            run "lm_fv_${n}" $(box 4) --physics lowmach --solver fv $a
        done ;;
    esac
done
