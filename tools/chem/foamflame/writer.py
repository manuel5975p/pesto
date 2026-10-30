"""OpenFOAM v2606 reactingFoam case writer for the C0 flame model.

Writes an ordinary case; runs nothing. Every model entry below was checked
against the v2606 SOURCE (paths in docs/chem.md "Against OpenFOAM
reactingFoam"):

- F -> P irreversibleArrheniusReaction, beta 0: StandardChemistryModel::omega
  = kf c_F, RR_F = -W kf rho Y_F / W = -A rho Y_F exp(-Ta/T) (floors c at 0,
  as dns floors Y).
- hConst + perfectGas + const transport, equal W and Cp for every specie:
  R, cp, mu, Pr do not depend on composition; Hs is the same function for F
  and P, so species diffusion carries no enthalpy (as dns). Qdot = -sum Hc RR
  = Hf_F omega with Hf_P = 0, Hf_F = q.
- Le = 1: reactingFoam's YEqn diffuses Y with muEff (Sc = 1, NOT unity Lewis;
  applications/solvers/combustion/reactingFoam/YEqn.H), while h diffuses with
  alphaEff = kappa/cp = mu/Pr. The case adds the difference (alphaEff - muEff)
  as an implicit laplacian fvOption on every solved specie, so rho D =
  kappa/cp exactly (`le1` below; off = the stock Sc = 1 solver).
- integrateReactionRate false: the instantaneous rate at the current PIMPLE
  iterate (laminar::correct -> calculate()), so converged outer correctors
  are implicit Euler on the whole system and a steady flame does not carry
  an operator-splitting error that depends on dt.
"""
import os

import numpy as np

HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def g(x):
    return f"{x:.15g}"


def put(case, rel, body, cls="dictionary"):
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HDR.format(cls=cls, obj=os.path.basename(rel)) + body)


def _list(vals, vector=False):
    n = len(vals)
    if vector:
        body = "\n".join("(" + " ".join(g(c) for c in v) + ")" for v in vals)
        return f"nonuniform List<vector>\n{n}\n(\n{body}\n)"
    return f"nonuniform List<scalar>\n{n}\n(\n" + "\n".join(g(v) for v in vals) + "\n)"


# --- mesh ---------------------------------------------------------------------

def block_mesh_1d(case, nx, h, ny=1):
    """One row of nx cubic cells of side h [m] (ny rows in y; z one cell),
    x in [0, nx h]: patches inlet (x lo), outlet (x hi), frontAndBack (empty)."""
    L, H, Z = nx * h, ny * h, h
    pts = [(0, 0, 0), (L, 0, 0), (L, H, 0), (0, H, 0),
           (0, 0, Z), (L, 0, Z), (L, H, Z), (0, H, Z)]
    body = "convertToMeters 1;\n\nvertices\n(\n" + "\n".join(
        "    (" + " ".join(g(c) for c in p) + ")" for p in pts) + "\n);\n\n"
    body += f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)\n);\n\n"
    body += """boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    frontAndBack
    {
        type empty;
        faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7));
    }
);
"""
    put(case, "system/blockMeshDict", body)


def block_mesh_2d(case, nx, ny, h):
    """nx x ny square cells of side h [m] (z one cell, empty), x in [0, nx h],
    y in [0, ny h]: inlet (x lo), outlet (x hi), the y faces a cyclic pair
    (dns' periodic y), frontAndBack empty."""
    L, H, Z = nx * h, ny * h, h
    pts = [(0, 0, 0), (L, 0, 0), (L, H, 0), (0, H, 0),
           (0, 0, Z), (L, 0, Z), (L, H, Z), (0, H, Z)]
    body = "convertToMeters 1;\n\nvertices\n(\n" + "\n".join(
        "    (" + " ".join(g(c) for c in p) + ")" for p in pts) + "\n);\n\n"
    body += f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)\n);\n\n"
    body += """boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    lower  { type cyclic; neighbourPatch upper; faces ((0 1 5 4)); }
    upper  { type cyclic; neighbourPatch lower; faces ((3 7 6 2)); }
    frontAndBack
    {
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }
);
"""
    put(case, "system/blockMeshDict", body)


def decompose(case, n):
    """decomposeParDict: scotch over n ranks."""
    put(case, "system/decomposeParDict",
        f"numberOfSubdomains {n};\nmethod scotch;\n")


# --- constant/ ----------------------------------------------------------------

def thermo_files(case, si, species, hf, thermo_type="hePsiThermo"):
    """thermophysicalProperties + thermo.foam: every specie of molWeight
    W, Cp = 3.5 R, Hf from `hf` (J/kg), const mu and Pr. thermo_type
    hePsiThermo (reactingFoam) | heRhoThermo (rhoReactingBuoyantFoam)."""
    put(case, "constant/thermophysicalProperties", f"""thermoType
{{
    type            {thermo_type};
    mixture         reactingMixture;
    transport       const;
    thermo          hConst;
    energy          sensibleEnthalpy;
    equationOfState perfectGas;
    specie          specie;
}}

inertSpecie         {species[-1]};
chemistryReader     foamChemistryReader;
foamChemistryFile   "<constant>/reactions";
foamChemistryThermoFile "<constant>/thermo.foam";
""")
    body = ""
    for s in species:
        body += f"""{s}
{{
    specie         {{ molWeight {g(si["W"])}; }}
    elements       {{ N 2; }}
    thermodynamics {{ Cp {g(si["cp"])}; Hf {g(hf.get(s, 0.0))}; }}
    transport      {{ mu {g(si["mu"])}; Pr {g(si["Pr"])}; }}
}}

"""
    put(case, "constant/thermo.foam", body)


def reactions_file(case, species, reactions):
    """reactions: list of (equation string, A [SI], Ta [K])."""
    body = "elements (N);\n\nspecies\n(\n" + "\n".join(
        f"    {s}" for s in species) + "\n);\n\nreactions\n{\n"
    for k, (eq, A, Ta) in enumerate(reactions):
        body += f"""    r{k}
    {{
        type     irreversibleArrheniusReaction;
        reaction "{eq}";
        A        {g(A)};
        beta     0;
        Ta       {g(Ta)};
    }}
"""
    put(case, "constant/reactions", body + "}\n")


def chemistry_files(case, integrate=False):
    put(case, "constant/chemistryProperties", """chemistryType
{
    solver          ode;
    method          standard;
}

chemistry           on;
initialChemicalTimeStep 1e-10;

odeCoeffs
{
    solver          seulex;
    absTol          1e-14;
    relTol          1e-8;
}
""")
    put(case, "constant/combustionProperties", f"""combustionModel laminar;
active          true;

laminarCoeffs
{{
    integrateReactionRate {"true" if integrate else "false"};
}}
""")
    put(case, "constant/turbulenceProperties", "simulationType laminar;\n")


# The Le = 1 correction: + laplacian(alphaEff - muEff, Y) on every solved
# specie (YEqn.H diffuses with muEff only). Implicit, positive for Pr < 1.
LE1 = r"""unityLewis
{
    type            scalarCodedSource;
    selectionMode   all;
    fields          (%(fields)s);
    name            unityLewis;
    codeInclude
    #{
        #include "turbulentFluidThermoModel.H"
        #include "fvmLaplacian.H"
    #};
    codeOptions
    #{
        -I$(LIB_SRC)/finiteVolume/lnInclude \
        -I$(LIB_SRC)/meshTools/lnInclude \
        -I$(LIB_SRC)/TurbulenceModels/turbulenceModels/lnInclude \
        -I$(LIB_SRC)/TurbulenceModels/compressible/lnInclude \
        -I$(LIB_SRC)/thermophysicalModels/basic/lnInclude \
        -I$(LIB_SRC)/transportModels/compressible/lnInclude
    #};
    codeLibs
    #{
        -lfiniteVolume -lmeshTools -lturbulenceModels -lcompressibleTurbulenceModels -lfluidThermophysicalModels
    #};
    codeAddSup
    #{
        FatalErrorInFunction << "unityLewis: rho form only" << exit(FatalError);
    #};
    codeAddSupRho
    #{
        const auto& turb = mesh_.lookupObject<compressible::turbulenceModel>(turbulenceModel::propertiesName);
        const volScalarField dGamma("dGammaLe1", turb.alphaEff() - turb.muEff());
        eqn += fvm::laplacian(dGamma, eqn.psi());
    #};
    codeCorrect #{ #};
    codeConstrain #{ #};
}
"""


# The gravity work on the sensible enthalpy: rhoReactingBuoyantFoam takes
# reactingFoam's EEqn.H, which has NO rho (U & g) (buoyantPimpleFoam's has);
# dns' E gains -g m_x (addGravity). Added here as a source on h.
BUOY = r"""buoyancyWork
{
    type            scalarCodedSource;
    selectionMode   all;
    fields          (h);
    name            buoyancyWork;
    codeInclude
    #{
        #include "gravityMeshObject.H"
    #};
    codeAddSup
    #{
        FatalErrorInFunction << "buoyancyWork: rho form only" << exit(FatalError);
    #};
    codeAddSupRho
    #{
        // v2606: g is a meshObjects::gravity on the Time registry
        const auto& g = meshObjects::gravity::New(mesh_.time());
        const auto& U = mesh_.lookupObject<volVectorField>("U");
        eqn += rho*(U & g);
    #};
    codeCorrect #{ #};
    codeConstrain #{ #};
}
"""


def fv_options(case, solved, le1=True, buoyancy_work=False):
    body = (LE1 % {"fields": " ".join(solved)} if le1 else "") + \
        (BUOY if buoyancy_work else "")
    if body:
        put(case, "system/fvOptions", body)


def gravity(case, g_si):
    """constant/g: g along -x (dns' candle axis), hRef 0 (gh = g.C)."""
    put(case, "constant/g", f"dimensions      [0 1 -2 0 0 0 0];\n"
        f"value           ({g(-g_si)} 0 0);\n", "uniformDimensionedVectorField")
    put(case, "constant/hRef", "dimensions      [0 1 0 0 0 0 0];\n"
        "value           0;\n", "uniformDimensionedScalarField")


# --- system/ ------------------------------------------------------------------

def schemes(case, yscheme, hscheme, solved, ddt="Euler",
            mv="multivariateIndependent", per=None, flux_required=None):
    """U, K central (Gauss linear); species and h through
    multivariateIndependent (each field its OWN scheme: multivariateSelection
    and plain multivariate share ONE limiter, the min over all fields, and
    apply it to EVERY field of the table, h included). `per`: {field:
    scheme} overriding yscheme / hscheme field by field (`blended 2`: a
    constant limiter 2 >= MC's, i.e. a field that does not vote in
    multivariateSelection's min)."""
    per = per or {}
    sel = " ".join(f"{s} {per.get(s, yscheme)};" for s in solved) + \
        f" h {per.get('h', hscheme)};"
    put(case, "system/fvSchemes", f"""ddtSchemes      {{ default {ddt}; }}
gradSchemes     {{ default Gauss linear; }}
divSchemes
{{
    default         none;
    div(phi,U)      Gauss linear;
    div(phi,Yi_h)   Gauss {mv} {{ {sel} }};
    div(phi,K)      Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default corrected; }}
""" + ("" if flux_required is None else
       f"fluxRequired    {{ default no; {flux_required}; }}\n"))


def solution(case, n_outer, n_corr, multi_d=False, p_name="p"):
    """1D: every solve to 1e-14 (DIC/DILU exact on a tridiagonal row).
    multi_d (2D cases): p by GAMG (GaussSeidel) to 1e-9 Pa absolute (p ~
    1e5 Pa: 1e-14 relative), U, h, Y by PBiCGStab/DILU to 1e-12; relTol 0
    on every corrector, so each PIMPLE pass is a converged linear solve.
    p_name "p_rgh": the buoyant solver's pressure (same solver)."""
    if multi_d:
        ps = ("{ solver GAMG; smoother GaussSeidel; tolerance 1e-9; "
              "relTol 0; }")
        us = ("{ solver PBiCGStab; preconditioner DILU; tolerance 1e-12; "
              "relTol 0; }")
    else:
        ps = "{ solver PCG; preconditioner DIC; tolerance 1e-14; relTol 0; }"
        us = ("{ solver PBiCGStab; preconditioner DILU; tolerance 1e-14; "
              "relTol 0; }")
    put(case, "system/fvSolution", f"""solvers
{{
    "rho.*"         {{ solver diagonal; }}
    {p_name}{" " * max(1, 16 - len(p_name))}{ps}
    {p_name}Final{" " * max(1, 11 - len(p_name))}{{ ${p_name}; }}
    "(U|h|Yi)"      {us}
    "(U|h|Yi)Final" {{ $U; }}
}}

PIMPLE
{{
    momentumPredictor yes;
    nOuterCorrectors  {n_outer};
    nCorrectors       {n_corr};
    nNonOrthogonalCorrectors 0;
}}
""")


def control(case, dt, t_end, write_dt, probes, burn_every=10, precision=14,
            application="reactingFoam", start_time=0.0):
    """Fixed dt (adjustTimeStep no); fields every write_dt; the burn
    integral (volIntegrate Qdot) and the probes (none: no probes object)
    every burn_every steps."""
    plocs = "\n".join("            (" + " ".join(g(c) for c in p) + ")"
                      for p in probes)
    ref = "" if not probes else f"""    ref
    {{
        type            probes;
        libs            (sampling);
        fields          (rho p T F U);
        probeLocations
        (
{plocs}
        );
        writeControl    timeStep;
        writeInterval   {burn_every};
    }}
"""
    put(case, "system/controlDict", f"""application     {application};
startFrom       startTime;
startTime       {g(start_time)};
stopAt          endTime;
endTime         {g(t_end)};
deltaT          {g(dt)};
adjustTimeStep  no;
writeControl    adjustableRunTime;
writeInterval   {g(write_dt)};
purgeWrite      0;
writeFormat     ascii;
writePrecision  {precision};
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;

functions
{{
    burn
    {{
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        fields          (Qdot);
        operation       volIntegrate;
        regionType      all;
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   {burn_every};
    }}
{ref}}}
""")


# --- 0/ -----------------------------------------------------------------------

def fields_0(case, ic, si, bcs, solved, species):
    """ic: dict of cell arrays in SI (T, p, U (n x 3), Y per specie).
    bcs: patch -> {field: dict text}. The inert specie (species[-1]) is
    written too (reactingFoam reads every specie)."""
    def field(name, cls, dims, internal):
        b = "".join(f"    {pt}\n    {{\n{pbc[name]}    }}\n"
                    for pt, pbc in bcs.items())
        b += "    frontAndBack { type empty; }\n"
        put(case, f"0/{name}", f"dimensions      {dims};\n\ninternalField   "
            f"{internal};\n\nboundaryField\n{{\n{b}}}\n", cls)
    field("T", "volScalarField", "[0 0 0 1 0 0 0]", _list(ic["T"]))
    field("p", "volScalarField", "[1 -1 -2 0 0 0 0]", _list(ic["p"]))
    if "p_rgh" in ic:                 # the buoyant solver's pressure
        field("p_rgh", "volScalarField", "[1 -1 -2 0 0 0 0]", _list(ic["p_rgh"]))
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", _list(ic["U"], True))
    for s in species:
        field(s, "volScalarField", "[0 0 0 0 0 0 0]", _list(ic[s]))


def bc(kind, value=None, **kw):
    """Patch-field dict text for fields_0. fixedValue `value`: a scalar, a
    3-vector (uniform), or a per-face array (n,) / (n, 3) (nonuniform, in the
    patch's face order)."""
    if kind == "fixedValue" and np.ndim(value) == 2 or (
            kind == "fixedValue" and np.ndim(value) == 1 and len(value) != 3):
        a = np.asarray(value, float)
        return (f"        type fixedValue;\n        value "
                f"{_list(a, vector=a.ndim == 2)};\n")
    if kind == "fixedValue":
        v = value if isinstance(value, str) else (
            "(" + " ".join(g(c) for c in value) + ")"
            if np.ndim(value) else g(value))
        return f"        type fixedValue;\n        value uniform {v};\n"
    if kind == "cyclic":
        return "        type cyclic;\n"
    if kind == "zeroGradient":
        return "        type zeroGradient;\n"
    if kind == "waveTransmissive":
        return (f"        type waveTransmissive;\n        gamma {g(kw['gamma'])};\n"
                f"        fieldInf {g(kw['pInf'])};\n        lInf {g(kw['lInf'])};\n"
                f"        value uniform {g(kw['pInf'])};\n")
    # the buoyant (candle) conditions: value a scalar or a 3-vector
    vtxt = None if value is None else (
        "(" + " ".join(g(c) for c in value) + ")" if np.ndim(value) else g(value))
    if kind in ("calculated", "fixedFluxPressure", "pressureInletOutletVelocity"):
        return f"        type {kind};\n        value uniform {vtxt};\n"
    if kind == "inletOutlet":         # value = the inflow (backflow) value
        return (f"        type inletOutlet;\n        inletValue uniform {vtxt};\n"
                f"        value uniform {vtxt};\n")
    if kind == "prghPressure":        # value = static p; rgh = the p_rgh start
        return (f"        type prghPressure;\n        p uniform {vtxt};\n"
                f"        value uniform {g(kw['rgh'])};\n")
    raise ValueError(kind)


# --- reading ------------------------------------------------------------------

def read_scalar_field(path):
    """Internal field of an ascii volScalarField (uniform or nonuniform)."""
    with open(path) as f:
        txt = f.read()
    k = txt.index("internalField")
    head = txt[k:k + 200]
    if "nonuniform" not in head:
        return float(head.split("uniform")[1].split(";")[0])
    a = txt.index("(", k)
    n = int(txt[k:a].split()[-1])
    b = txt.index(")", a)
    v = np.array(txt[a + 1:b].split(), float)
    assert len(v) == n
    return v


def read_vector(path):
    """Internal field of an ascii nonuniform volVectorField, (n, 3)."""
    with open(path) as f:
        txt = f.read()
    k = txt.index("internalField")
    a = txt.index("(", k)
    n = int(txt[k:a].split()[-1])
    b = txt.index("\n)", a)
    body = txt[a + 1:b].replace("(", " ").replace(")", " ")
    v = np.array(body.split(), float)
    assert len(v) == 3 * n
    return v.reshape(n, 3)


def read_vector_x(path):
    with open(path) as f:
        txt = f.read()
    k = txt.index("internalField")
    a = txt.index("(", k)
    n = int(txt[k:a].split()[-1])
    rows = txt[a + 1:].split("\n")
    out = []
    for r in rows:
        r = r.strip()
        if r.startswith("("):
            out.append(float(r[1:].split()[0]))
            if len(out) == n:
                break
    return np.array(out)
