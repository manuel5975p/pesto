/*---------------------------------------------------------------------------*\
  foamAmiDump: the reference of dns/fvami (tests/test_fvami.cpp, stage 3c).

  Reads a case, optionally rotates the points of a cellZone's cells by an
  angle about an axis (fvMesh::movePoints), and writes per cyclicAMI OWNER
  patch what its AMIInterpolation holds, as hex floats:

      srcMagSf srcWeightsSum srcAddress srcWeights
      tgtMagSf tgtWeightsSum tgtAddress tgtWeights
      interpolateToSource / interpolateToTarget of f(j) = 1 + j/4 + j^2/512
      (j = the other side's patch face index: exact in double)

  and with -fv per patch of the pair: Cf, Cn, nf, weights(), delta(),
  deltaCoeffs, nonOrthDeltaCoeffs, nonOrthCorrectionVectors, and
  cyclicAMIPolyPatch::interpolate of x + 2 y + 3 z at the neighbour's Cf.

  The rotation is written out in scalars (Rodrigues), the expression
  tests/test_fvami.cpp repeats, so both sides hold the same points bit for bit
  (the dump carries an FNV-1a hash of them and the rotation: zone, angle,
  axis, origin; zone `-` = none).

  usage: foamAmiDump -case CASE [-zone Z -angle RAD [-axis '(0 0 1)']
                     [-origin '(0 0 0)']] [-fv] [-out FILE]

  Build (OpenFOAM v2606 sourced; objects stay out of the repository):
      cp -r tools/fv/foamAmiDump runs/ami3c/build/ && cd runs/ami3c/build/foamAmiDump
      FOAM_USER_APPBIN=$PWD/../../bin wmake
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "cyclicAMIPolyPatch.H"
#include "cyclicAMIFvPatch.H"
#include "bitSet.H"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{

void putScalars(FILE* f, const char* name, const Foam::UList<Foam::scalar>& v)
{
    std::fprintf(f, "%s %d\n", name, int(v.size()));
    forAll(v, i)
    {
        std::fprintf(f, "%a\n", v[i]);
    }
}

void putVectors(FILE* f, const char* name, const Foam::UList<Foam::vector>& v)
{
    std::fprintf(f, "%s %d\n", name, int(v.size()));
    forAll(v, i)
    {
        std::fprintf(f, "%a %a %a\n", v[i].x(), v[i].y(), v[i].z());
    }
}

void putLabelLists(FILE* f, const char* name, const Foam::labelListList& v)
{
    std::fprintf(f, "%s %d\n", name, int(v.size()));
    forAll(v, i)
    {
        std::fprintf(f, "%d", int(v[i].size()));
        forAll(v[i], j)
        {
            std::fprintf(f, " %d", int(v[i][j]));
        }
        std::fprintf(f, "\n");
    }
}

void putScalarLists(FILE* f, const char* name, const Foam::scalarListList& v)
{
    std::fprintf(f, "%s %d\n", name, int(v.size()));
    forAll(v, i)
    {
        std::fprintf(f, "%d", int(v[i].size()));
        forAll(v[i], j)
        {
            std::fprintf(f, " %a", v[i][j]);
        }
        std::fprintf(f, "\n");
    }
}

// f(j) = 1 + j/4 + j^2/512
Foam::scalarField indexField(const Foam::label n)
{
    Foam::scalarField fld(n);
    forAll(fld, j)
    {
        const double x = double(j);
        fld[j] = 1.0 + 0.25*x + 0.001953125*x*x;
    }
    return fld;
}

} // End anonymous namespace


int main(int argc, char *argv[])
{
    argList::addNote("Dumps cyclicAMI addressing and weights (dns/fvami reference)");
    argList::addOption("zone", "name", "cellZone whose points are rotated");
    argList::addOption("angle", "rad", "rotation angle");
    argList::addOption("axis", "vector", "rotation axis, default (0 0 1)");
    argList::addOption("origin", "point", "rotation origin, default (0 0 0)");
    argList::addOption("out", "file", "output file, default CASE/amiDump");
    argList::addBoolOption("fv", "also dump the fvPatch quantities");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const fileName outName
    (
        args.getOrDefault<fileName>("out", runTime.path()/"amiDump")
    );
    const bool doFv = args.found("fv");

    scalar angle = 0;
    word zoneName("-");
    const vector axis(args.getOrDefault<vector>("axis", vector(0, 0, 1)));
    const point origin(args.getOrDefault<point>("origin", point::zero));
    if (args.readIfPresent("zone", zoneName))
    {
        angle = args.get<scalar>("angle");

        const label zonei = mesh.cellZones().findZoneID(zoneName);
        if (zonei < 0)
        {
            FatalErrorInFunction
                << "No cellZone " << zoneName << exit(FatalError);
        }

        bitSet movePts(mesh.nPoints());
        for (const label celli : mesh.cellZones()[zonei])
        {
            for (const label facei : mesh.cells()[celli])
            {
                movePts.set(mesh.faces()[facei]);
            }
        }

        // Rodrigues, scalar by scalar (tests/test_fvami.cpp repeats it)
        const double ax = axis.x(), ay = axis.y(), az = axis.z();
        const double m = std::sqrt(ax*ax + ay*ay + az*az);
        const double kx = ax/m, ky = ay/m, kz = az/m;
        const double c = std::cos(double(angle)), s = std::sin(double(angle));
        const double ox = origin.x(), oy = origin.y(), oz = origin.z();

        pointField newPoints(mesh.points());
        forAll(newPoints, pointi)
        {
            if (!movePts.test(pointi)) continue;

            const double dx = newPoints[pointi].x() - ox;
            const double dy = newPoints[pointi].y() - oy;
            const double dz = newPoints[pointi].z() - oz;
            const double kd = kx*dx + ky*dy + kz*dz;
            const double cx = ky*dz - kz*dy;
            const double cy = kz*dx - kx*dz;
            const double cz = kx*dy - ky*dx;
            const double rx = dx*c + cx*s + kx*kd*(1.0 - c);
            const double ry = dy*c + cy*s + ky*kd*(1.0 - c);
            const double rz = dz*c + cz*s + kz*kd*(1.0 - c);
            newPoints[pointi] = point(ox + rx, oy + ry, oz + rz);
        }

        Info<< "Rotating " << movePts.count() << " points of zone "
            << zoneName << " by " << angle << " rad" << endl;

        mesh.movePoints(newPoints);
    }

    FILE* f = std::fopen(outName.c_str(), "w");
    if (!f)
    {
        FatalErrorInFunction
            << "Cannot open " << outName << exit(FatalError);
    }

    // FNV-1a over the points' bytes
    std::uint64_t hash = 1469598103934665603ULL;
    {
        const pointField& pts = mesh.points();
        forAll(pts, pointi)
        {
            const double xyz[3] = {pts[pointi].x(), pts[pointi].y(), pts[pointi].z()};
            unsigned char b[sizeof(xyz)];
            std::memcpy(b, xyz, sizeof(xyz));
            for (unsigned char ch : b)
            {
                hash ^= ch;
                hash *= 1099511628211ULL;
            }
        }
    }
    std::fprintf(f, "foamAmiDump 1\n");
    std::fprintf
    (
        f, "points %d hash %016llx\n",
        int(mesh.nPoints()), static_cast<unsigned long long>(hash)
    );
    std::fprintf
    (
        f, "rotate %s %a %a %a %a %a %a %a\n",
        zoneName.c_str(), double(angle),
        axis.x(), axis.y(), axis.z(), origin.x(), origin.y(), origin.z()
    );

    const polyBoundaryMesh& pbm = mesh.boundaryMesh();

    forAll(pbm, patchi)
    {
        const auto* cppPtr = isA<cyclicAMIPolyPatch>(pbm[patchi]);
        if (!cppPtr || !cppPtr->owner()) continue;

        const cyclicAMIPolyPatch& pp = *cppPtr;
        const cyclicAMIPolyPatch& nbr = pp.neighbPatch();
        const AMIPatchToPatchInterpolation& ami = pp.AMI();

        std::fprintf
        (
            f, "pair %s %s %d %d\n",
            pp.name().c_str(), nbr.name().c_str(),
            int(pp.size()), int(nbr.size())
        );

        putScalars(f, "srcMagSf", ami.srcMagSf());
        putScalars(f, "srcWeightsSum", ami.srcWeightsSum());
        putLabelLists(f, "srcAddress", ami.srcAddress());
        putScalarLists(f, "srcWeights", ami.srcWeights());
        putScalars(f, "tgtMagSf", ami.tgtMagSf());
        putScalars(f, "tgtWeightsSum", ami.tgtWeightsSum());
        putLabelLists(f, "tgtAddress", ami.tgtAddress());
        putScalarLists(f, "tgtWeights", ami.tgtWeights());

        {
            const scalarField fldT(indexField(nbr.size()));
            const scalarField fldS(indexField(pp.size()));
            const scalarField toSrc(ami.interpolateToSource(fldT));
            const scalarField toTgt(ami.interpolateToTarget(fldS));
            putScalars(f, "interpolateToSource", toSrc);
            putScalars(f, "interpolateToTarget", toTgt);
        }

        if (doFv)
        {
            const label ids[2] = {pp.index(), nbr.index()};
            for (const label pi : ids)
            {
                const fvPatch& fp = mesh.boundary()[pi];
                const auto& cpp = refCast<const cyclicAMIPolyPatch>(pbm[pi]);
                const fvPatch& nfp = mesh.boundary()[cpp.neighbPatchID()];

                std::fprintf(f, "fvPatch %s %d\n", fp.name().c_str(), int(fp.size()));
                putVectors(f, "Cf", fp.Cf());
                putVectors(f, "Cn", vectorField(fp.Cn()));
                putVectors(f, "nf", vectorField(fp.nf()));
                putScalars(f, "weights", mesh.weights().boundaryField()[pi]);
                putVectors(f, "delta", vectorField(fp.delta()));
                putScalars(f, "deltaCoeffs", mesh.deltaCoeffs().boundaryField()[pi]);
                putScalars
                (
                    f, "nonOrthDeltaCoeffs",
                    mesh.nonOrthDeltaCoeffs().boundaryField()[pi]
                );
                putVectors
                (
                    f, "nonOrthCorrectionVectors",
                    mesh.nonOrthCorrectionVectors().boundaryField()[pi]
                );

                // x + 2 y + 3 z at the neighbour's face centres
                const vectorField& nCf = nfp.Cf();
                scalarField lin(nCf.size());
                forAll(lin, i)
                {
                    lin[i] = nCf[i].x() + 2.0*nCf[i].y() + 3.0*nCf[i].z();
                }
                const scalarField linI(cpp.interpolate(lin));
                putScalars(f, "interpolateLinear", linI);
            }
        }
    }

    std::fprintf(f, "end\n");
    std::fclose(f);

    Info<< "Wrote " << outName << nl << "End" << nl << endl;

    return 0;
}


// ************************************************************************* //
