"""
Improved Point Sampling (IPS) point generator for CICYs.

IPS samples points with several Fubini-Study metrics, each concentrating on the
part of the manifold the previous ones under-sampled, and weights them so that
Monte Carlo integrals stay unbiased (Keller and Lukic, arXiv:0907.1387, Sec. 3.2).
The sampling itself runs in Mathematica (``PointGeneratorMathematicaCICYIPS.m``),
driven from Python through wolframclient.

Used like the other point generators:

>>> pg = IPSPointGeneratorMathematica(monomials, coefficients, kmoduli, ambient)
>>> kappa = pg.prepare_dataset(n_p, dirname, num_regions=5)
>>> pg.prepare_basis(dirname, kappa)

which writes ``dataset.npz`` and ``basis.pickle``. Pass ``write_raw=True`` to
also keep the raw IPS diagnostics (per-point CSV files, the metrics, region
labels and acceptance rates).

:Authors:
    Eric Habjan (IPS implementation)
    Fabian Ruehle f.ruehle@northeastern.edu (cymetric integration)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from wolframclient.deserializers import binary_deserialize
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import Global as wlGlobal
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export as wlexport

from cymetric.pointgen.pointgen_mathematica import (
    ComplexFunctionConsumer,
    PointGeneratorMathematica,
    _resolve_kernel_path,
)


# Lives next to the other Mathematica sources so that package_data ships it.
IPS_M_FILE = str(Path(__file__).resolve().parent.parent / "wolfram"
                 / "PointGeneratorMathematicaCICYIPS.m")

logger = logging.getLogger("pointgenIPS_mathematica")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(name)s:%(levelname)s:%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _patch_globals_from_local(patch_local: np.ndarray, ambient: np.ndarray) -> np.ndarray:
    """Mirror Mathematica's ``patchGlobalsFromLocal``.

    ``patch_local`` is an ``(npts, n_blocks)`` 1-indexed array of local patch
    indices; ``ambient`` is the projective dimensions vector. Returns
    ``(npts, n_blocks)`` of 1-indexed flattened global coordinate indices.
    """
    ambient = np.asarray(ambient, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(ambient + 1)[:-1]])
    return patch_local + offsets[None, :]


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr[:, None] if arr.ndim == 1 else arr


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class IPSPointGeneratorMathematica(PointGeneratorMathematica):
    """IPS point generator backed by ``GeneratePointsMCICYIPS`` in Mathematica.

    Inherits from cymetric's ``PointGeneratorMathematica`` so the standard
    cymetric machinery (``pullbacks``, ``fubini_study_metrics``,
    ``prepare_basis``) works unchanged. The ``generate_ips`` method drives the
    IPS sampler over a ``WolframLanguageSession`` and stores the full result
    set as ``ips_*`` attributes. ``prepare_dataset`` runs sampling and writes
    every artifact downstream code needs.
    """

    # Weights returned by the Mathematica sampler carry the IPS region
    # rejection/ownership acceptance-rate correction, but NOT cymetric's
    # Kaehler-volume (vol_j) normalization. When normalize_to_vol_j=True the
    # weights are rescaled in Python (see vol_j_normalization_factor) so that
    # \int_X det(g) = kappa_{ijk} t^i t^j t^k, mirroring cymetric's
    # point_weight. Defaults to True, as for the other point generators.
    # Single source of truth for the generate_point_weights /
    # prepare_dataset signature defaults and the value recorded in metadata,
    # so they cannot drift apart.
    DEFAULT_NORMALIZE_TO_VOL_J: bool = True

    def __init__(
        self,
        monomials,
        coefficients,
        kmoduli,
        ambient,
        precision: int = 20,
        verbose: int = 1,
        vol_j_norm: Optional[float] = None,
        kernel_path: Optional[str] = None,
        kappa_sample_size: int = 25000,
    ):
        super().__init__(
            monomials,
            coefficients,
            kmoduli,
            ambient,
            precision=precision,
            verbose=verbose,
            vol_j_norm=vol_j_norm,
            kernel_path=kernel_path,
        )

        # Size of the fresh, independent sample used to estimate each metric's
        # kappa (see EstimateKappaCICY in PointGeneratorMathematicaCICYIPS.m).
        self.kappa_sample_size: int = int(kappa_sample_size)

        # Populated by generate_ips
        self.ips_points: Optional[np.ndarray] = None
        self.ips_weights: Optional[np.ndarray] = None
        # Mathematica returns the COMPLEX holomorphic volume-form factor
        # Omega = 1/det(B), matching the convention of the other point
        # generators. |Omega|^2 is derived in Python where needed (see
        # ips_abs_omega_squared()); never store |Omega| or |Omega|^2 here.
        self.ips_omega: Optional[np.ndarray] = None
        self.ips_patches_local: Optional[np.ndarray] = None   # 1-indexed
        self.ips_patches_global: Optional[np.ndarray] = None  # 1-indexed
        self.ips_j_elim_global: Optional[np.ndarray] = None   # 1-indexed
        self.ips_region_labels: Optional[np.ndarray] = None   # 1-indexed
        self.ips_kappas: Optional[np.ndarray] = None
        self.ips_acceptances: Optional[np.ndarray] = None
        self.ips_num_samples: Optional[np.ndarray] = None
        self.ips_dim_cy: Optional[int] = None
        self.ips_Ls: Optional[list] = None
        self.ips_total_pts_requested: Optional[int] = None
        self.ips_num_regions: Optional[int] = None
        self.ips_front_end: bool = False
        # Cached cymetric-style vol_j normalization factor for the current
        # ips_* arrays; invalidated whenever they change.
        self._ips_vol_j_norm_fac: Optional[float] = None

    # ----- sampling --------------------------------------------------------

    def generate_ips(
        self,
        total_pts: int,
        num_regions: int,
        *,
        nproc: int = -1,
        front_end: bool = False,
    ) -> dict:
        """Call the IPS Mathematica sampler and parse its result.

        Populates ``self.ips_*`` attributes and returns a dict view of them.
        ``nproc=-1`` (default) uses ``SLURM_CPUS_PER_TASK`` if set, otherwise
        ``$ProcessorCount``.
        """
        if num_regions < 1:
            raise ValueError(f"num_regions must be >= 1; got {num_regions}")
        if total_pts < 1:
            raise ValueError(f"total_pts must be >= 1; got {total_pts}")

        self.ips_total_pts_requested = int(total_pts)
        self.ips_num_regions = int(num_regions)
        self.ips_front_end = bool(front_end)

        ambient_arg = self.ambient.tolist()
        coeffs_arg = [c.astype(np.complex128).tolist() for c in self.coefficients]
        monos_arg = [m.astype(np.int64).tolist() for m in self.monomials]
        kmoduli_arg = np.asarray(self.kmoduli, dtype=np.float64).tolist()

        resolved_kernel = _resolve_kernel_path(self.kernel_path)
        logger.info(f"Opening WolframLanguageSession "
                    f"(kernel={resolved_kernel or 'auto-detected'})")
        with WolframLanguageSession(kernel=resolved_kernel) as session:
            self.wl_session = session
            session.evaluate(wlexpr("ClientLibrary`SetErrorLogLevel[]"))

            logger.info(f"Loading IPS sampler: {IPS_M_FILE}")
            session.evaluate(wlexpr(f'Get["{IPS_M_FILE}"]'))

            self._launch_kernels(session, nproc)

            ips_fn = session.function(wlGlobal.GeneratePointsMCICYIPS)
            logger.info(
                f"Calling GeneratePointsMCICYIPS[total={total_pts}, regions={num_regions}, "
                f"precision={self.precision}, verbose={self.verbose}, frontEnd={front_end}, "
                f"kappaSampleSize={self.kappa_sample_size}]"
            )
            raw = ips_fn(
                int(total_pts),
                int(num_regions),
                ambient_arg,
                coeffs_arg,
                monos_arg,
                kmoduli_arg,
                int(self.precision),
                int(self.verbose),
                bool(front_end),
                int(self.kappa_sample_size),
            )

            wxf = wlexport(raw, target_format="wxf")
            parsed = binary_deserialize(wxf, consumer=ComplexFunctionConsumer())

        if not isinstance(parsed, (list, tuple)) or len(parsed) < 11:
            raise RuntimeError(
                f"Expected 11-tuple from GeneratePointsMCICYIPS; got {type(parsed).__name__} "
                f"of length {len(parsed) if hasattr(parsed, '__len__') else '?'}"
            )

        (
            points, weights, omegas, patches_local, j_elim_global,
            region_labels, kappas, acceptances, num_samples,
            dim_cy_box, Ls,
        ) = parsed[:11]

        self.ips_points = np.array(points, dtype=np.complex128)
        self.ips_weights = np.array(weights, dtype=np.float64).reshape(-1)
        # Third field from Mathematica is the complex Omega = 1/det(B).
        self.ips_omega = np.array(omegas, dtype=np.complex128).reshape(-1)
        self.ips_patches_local = _ensure_2d(np.array(patches_local, dtype=np.int64))
        self.ips_j_elim_global = _ensure_2d(np.array(j_elim_global, dtype=np.int64))
        self.ips_region_labels = np.array(region_labels, dtype=np.int64).reshape(-1)
        self.ips_kappas = np.array(kappas, dtype=np.float64).reshape(-1)
        self.ips_acceptances = np.array(acceptances, dtype=np.float64).reshape(-1)
        self.ips_num_samples = np.array(num_samples, dtype=np.float64).reshape(-1)
        self.ips_dim_cy = int(np.asarray(dim_cy_box).flatten()[0])
        self.ips_Ls = [
            [np.array(blk, dtype=np.complex128) for blk in region_blocks]
            for region_blocks in Ls
        ]
        self.ips_patches_global = _patch_globals_from_local(
            self.ips_patches_local, self.ambient
        )
        # The sampler returns the points grouped by region. Shuffle them, as
        # CICYPointGenerator does with its per-assignment samples, so that the
        # train/validation split and any leading subset (generate_points) see
        # every region in proportion; otherwise the validation set holds only
        # the last region and the training set under-weights it.
        self._take_rows(np.random.permutation(len(self.ips_points)))

        logger.info(
            f"IPS sampling returned {len(self.ips_points)} points "
            f"across {len(self.ips_kappas)} region(s)."
        )
        return {
            "points": self.ips_points,
            "weights": self.ips_weights,
            "omega": self.ips_omega,  # complex Omega = 1/det(B)
            "patches_local": self.ips_patches_local,
            "patches_global": self.ips_patches_global,
            "j_elim_global": self.ips_j_elim_global,
            "region_labels": self.ips_region_labels,
            "kappas": self.ips_kappas,
            "acceptances": self.ips_acceptances,
            "num_samples": self.ips_num_samples,
            "dim_cy": self.ips_dim_cy,
            "Ls": self.ips_Ls,
        }

    def _launch_kernels(self, session: WolframLanguageSession, nproc: int):
        if nproc is None or nproc < 0:
            slurm = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm:
                session.evaluate(wlexpr(f"LaunchKernels[{int(slurm)}]"))
                logger.info(f"Launched {int(slurm)} subkernels (SLURM_CPUS_PER_TASK).")
            else:
                session.evaluate(wlexpr("LaunchKernels[$ProcessorCount]"))
                n = session.evaluate(wlexpr("Length[Kernels[]]"))
                logger.info(f"Launched {n} subkernels ($ProcessorCount).")
        else:
            session.evaluate(wlexpr(f"LaunchKernels[{int(nproc)}]"))
            logger.info(f"Launched {int(nproc)} subkernels.")

    # ----- cymetric overrides ---------------------------------------------

    def generate_points(self, n_p, nproc=-1, **kwargs):
        """Return the first ``n_p`` IPS-sampled points."""
        if self.ips_points is None:
            raise RuntimeError(
                "No IPS data loaded. Call generate_ips(total_pts, num_regions) "
                "or prepare_dataset(...) first."
            )
        if n_p > len(self.ips_points):
            raise ValueError(
                f"Requested {n_p} points but only {len(self.ips_points)} sampled."
            )
        return self.ips_points[:n_p]

    def generate_point_weights(
        self, n_pw, omega=False, normalize_to_vol_j=DEFAULT_NORMALIZE_TO_VOL_J
    ):
        """Return IPS-precomputed weights/omegas as a structured array.

        ``normalize_to_vol_j`` defaults to ``True``, as for the other point
        generators. The Mathematica sampler applies the
        IPS region rejection/ownership acceptance-rate correction to the
        weights; that part is kept unchanged. When ``normalize_to_vol_j`` is
        True the weights are additionally rescaled here by the cymetric-style
        Kaehler-volume factor (see ``vol_j_normalization_factor``) so that
        ``\\int_X det(g) = kappa_{ijk} t^i t^j t^k``.
        """
        if self.ips_points is None:
            raise RuntimeError("No IPS data loaded; call generate_ips() first.")
        n = min(n_pw, len(self.ips_points))
        fields = [("point", np.complex128, self.ncoords), ("weight", np.float64)]
        if omega:
            fields.append(("omega", np.complex128))
        out = np.zeros(n, dtype=np.dtype(fields))
        out["point"] = self.ips_points[:n]
        out["weight"] = self.ips_weights[:n]
        if normalize_to_vol_j:
            out["weight"] *= self.vol_j_normalization_factor()
        if omega:
            # cymetric's interface contract: the field MUST be named "omega"
            # and hold the COMPLEX holomorphic volume-form factor Omega, as
            # for every other point generator. The Mathematica sampler now
            # returns Omega = 1/det(B) directly, so it is passed through
            # unchanged; cymetric.pointgen.nphelper.prepare_dataset then
            # computes np.real(omega * np.conj(omega)) -> |Omega|^2 itself.
            out["omega"] = self.ips_omega[:n]
        return out

    # ----- packaging -------------------------------------------------------

    def prepare_dataset(
        self,
        n_p: int,
        dirname: Union[str, Path],
        val_split: float = 0.1,
        ltails: float = 0,
        rtails: float = 0,
        *,
        num_regions: int = 1,
        shuffle: bool = False,
        seed: int = 2025,
        nproc: int = -1,
        front_end: bool = False,
        normalize_to_vol_j: bool = DEFAULT_NORMALIZE_TO_VOL_J,
        write_raw: bool = False,
    ) -> float:
        """Samples with IPS and writes the training data, like the other generators.

        The positional arguments are exactly those of every other cymetric point
        generator, ``prepare_dataset(n_p, dirname, val_split, ltails, rtails)``,
        and mean the same thing; the IPS-specific options are keyword-only. So

        >>> kappa = pg.prepare_dataset(10000, "./mypoints", num_regions=5)
        >>> pg.prepare_basis("./mypoints", kappa)

        samples 10,000 points over five IPS regions.

        Args:
            n_p (int): Number of points.
            dirname (str or Path): Output directory.
            val_split (float, optional): Fraction used for validation.
                Defaults to 0.1.
            ltails (float, optional): Fraction of lowest-weight points to
                discard. Defaults to 0.
            rtails (float, optional): Fraction of highest-weight points to
                discard. Defaults to 0. As for the other generators, the
                sample is enlarged by ``1/(1 - ltails - rtails)`` first, so
                about ``n_p`` points remain. Discarding points by weight
                changes the sampling distribution without a matching
                correction to the weights, so it biases integrals even for
                small values; leave both at 0 unless that is accounted for.
                Only supported for ``num_regions = 1``: trimming by pooled
                weight would also break the equal per-region counts the
                stratified weights rely on.
            num_regions (int, optional): Number of IPS regions (metrics).
                Defaults to 1.
            shuffle (bool, optional): Reshuffle before the train/validation
                split. Not needed for a representative split: the sample is
                already shuffled once right after sampling, as for
                CICYPointGenerator. Only supported for ``num_regions = 1``,
                since it breaks the region/point alignment of the raw outputs.
            seed (int, optional): Seed for the shuffle. Defaults to 2025.
            nproc (int, optional): Mathematica subkernels; -1 uses all.
            front_end (bool, optional): Print through the Mathematica front end.
            normalize_to_vol_j (bool, optional): Rescale the weights so that
                int_X det(g) equals the intersection number, as the other
                generators do. Defaults to True.
            write_raw (bool, optional): Also write the raw IPS diagnostics:
                per-point CSV files, ``L_matrices.json``, ``metadata.json``,
                ``mathematica_extras.npz``, ``pack_summary.json`` and
                ``point_gen.pickle``. Defaults to False, in which case only
                ``dataset.npz`` and ``basis.pickle`` are written, as for the
                other generators.

        Returns:
            float: kappa, computed from the exported data exactly as the other
            generators compute it.
        """
        num_regions = int(num_regions)
        if not (0 <= ltails < 1 and 0 <= rtails < 1 and ltails + rtails < 1):
            raise ValueError(f"need 0 <= ltails, rtails and ltails + rtails < 1; "
                             f"got ltails={ltails}, rtails={rtails}")
        if num_regions > 1 and (ltails or rtails):
            raise ValueError(
                "ltails/rtails are only supported for num_regions = 1: trimming "
                "by pooled weight breaks the equal per-region counts that the "
                "stratified IPS weights rely on.")
        if shuffle and num_regions > 1:
            raise ValueError(
                "shuffle=True is unsafe when num_regions > 1: it breaks "
                "region/point alignment relied on by region-weighted integration."
            )

        d = Path(dirname)
        d.mkdir(parents=True, exist_ok=True)

        # Sample unless a matching sample already exists (e.g. from an explicit
        # generate_ips call). A cached sample is never silently reused for a
        # different request.
        n_request = int(round(n_p / (1.0 - ltails - rtails)))
        if (self.ips_points is None
                or self.ips_num_regions != num_regions
                or self.ips_total_pts_requested != n_request):
            self.generate_ips(n_request, num_regions, nproc=nproc, front_end=front_end)

        # Recorded so metadata.json reflects the flag actually in effect.
        self.ips_normalize_to_vol_j = bool(normalize_to_vol_j)

        self._filter_invalid_rows()
        if ltails or rtails:
            w = np.sort(self.ips_weights)
            lower = w[int(round(ltails * len(w)))]
            upper = w[int(round((1 - rtails) * len(w))) - 1]
            self._take_rows((self.ips_weights >= lower) & (self.ips_weights <= upper))

        if write_raw:
            self._write_raw_outputs(d)
        return self._write_cymetric_outputs(
            d, val_split=val_split, shuffle=shuffle, seed=seed,
            normalize_to_vol_j=normalize_to_vol_j, write_raw=write_raw,
        )

    # ----- internals -------------------------------------------------------

    def ips_abs_omega_squared(self) -> np.ndarray:
        """|Omega|^2 derived from the stored complex Omega.

        Computed exactly as cymetric's ``nphelper.prepare_dataset`` does:
        ``np.real(omega * np.conj(omega))``. This is the only sanctioned way
        to obtain |Omega|^2 from the IPS data; the stored ``ips_omega`` itself
        is always the complex Omega = 1/det(B).
        """
        if self.ips_omega is None:
            raise RuntimeError("No IPS data loaded; call generate_ips() first.")
        return np.real(self.ips_omega * np.conj(self.ips_omega))

    def vol_j_normalization_factor(self) -> float:
        r"""Cymetric-style Kaehler-volume normalization factor for IPS weights.

        The Mathematica sampler's weights already include the IPS
        acceptance-rate correction, but not cymetric's ``normalize_to_vol_j``
        rescaling (cymetric's ``point_weight``). This computes the constant
        ``norm_fac`` such that the rescaled weights ``norm_fac * w`` satisfy

        .. math::

            \int_X \det(g) = \frac{1}{N}\sum_i
                \det(g^\text{ref}_\text{pb})\,
                \frac{\text{norm\_fac}\, w}{|\Omega|^2}\Big|_{x_i}
            = \kappa_{ijk} t^i t^j t^k,

        where :math:`g^\text{ref}` is the reference Fubini-Study metric at the
        Kaehler moduli ``t = kmoduli`` pulled back to the CY, and the effective
        sampling determinant is recovered from the IPS data as
        ``det_eff = |Omega|^2 / w`` (so ``mean(det_ref / det_eff)`` is exactly
        cymetric's ``mean(det(fs_ref_pb) / detg_norm)``). The pullbacks use the
        sampler's own eliminated coordinates so ``det_ref`` and ``det_eff``
        live in the same local coordinates.

        The factor is computed once from the full current point set and
        cached; the cache is invalidated when the ``ips_*`` arrays change.
        """
        fac = getattr(self, "_ips_vol_j_norm_fac", None)
        if fac is not None:
            return fac
        if self.ips_points is None:
            raise RuntimeError("No IPS data loaded; call generate_ips() first.")
        pbs = self.pullbacks(self.ips_points, j_elim=self.ips_j_elim_global - 1)
        fs_ref = self.fubini_study_metrics(self.ips_points, vol_js=self.kmoduli)
        fs_ref_pb = np.einsum("xai,xij,xbj->xab", pbs, fs_ref, np.conj(pbs))
        det_ref = np.real(np.linalg.det(fs_ref_pb))
        det_eff = self.ips_abs_omega_squared() / self.ips_weights
        target_vol = float(self.get_volume_from_intersections(self.kmoduli))
        fac = target_vol / float(np.mean(det_ref / det_eff))
        logger.info(
            f"vol_j normalization: target volume kappa_ijk t^i t^j t^k = "
            f"{target_vol:.6e}, factor = {fac:.6e}"
        )
        self._ips_vol_j_norm_fac = fac
        return fac

    def _filter_invalid_rows(self):
        p = self.ips_points
        mask = (
            np.all(np.isfinite(p.real), axis=1)
            & np.all(np.isfinite(p.imag), axis=1)
            & np.isfinite(self.ips_weights)
            & np.isfinite(self.ips_omega.real)
            & np.isfinite(self.ips_omega.imag)
            & (self.ips_weights > 0)
        )
        if np.all(mask):
            return
        n_bad = int(np.count_nonzero(~mask))
        logger.warning(f"Filtering {n_bad} invalid IPS rows before export.")
        self._take_rows(mask)

    def _take_rows(self, rows):
        """Keeps or reorders the rows consistently across every per-point array.

        ``rows`` is a boolean mask or an index array, as in numpy indexing.
        """
        self._ips_vol_j_norm_fac = None
        self.ips_points = self.ips_points[rows]
        self.ips_weights = self.ips_weights[rows]
        self.ips_omega = self.ips_omega[rows]
        self.ips_patches_local = self.ips_patches_local[rows]
        self.ips_patches_global = self.ips_patches_global[rows]
        self.ips_j_elim_global = self.ips_j_elim_global[rows]
        self.ips_region_labels = self.ips_region_labels[rows]

    def _write_raw_outputs(self, d: Path):
        np.savetxt(d / "points_real.csv", self.ips_points.real, delimiter=",")
        np.savetxt(d / "points_imag.csv", self.ips_points.imag, delimiter=",")
        np.savetxt(d / "weights.csv", self.ips_weights, delimiter=",")
        # Complex Omega = 1/det(B) split into real/imag parts, plus the derived
        # |Omega|^2 for convenience/backwards compatibility. Each filename says
        # exactly what it holds; nothing named "omega" ever stores a magnitude.
        np.savetxt(d / "omega_real.csv", self.ips_omega.real, delimiter=",")
        np.savetxt(d / "omega_imag.csv", self.ips_omega.imag, delimiter=",")
        np.savetxt(d / "omega_squared.csv", self.ips_abs_omega_squared(), delimiter=",")
        np.savetxt(d / "kappas.csv", self.ips_kappas, delimiter=",")
        np.savetxt(d / "patches_local.csv", self.ips_patches_local, delimiter=",", fmt="%d")
        np.savetxt(d / "patches_global.csv", self.ips_patches_global, delimiter=",", fmt="%d")
        np.savetxt(d / "j_elim_global.csv", self.ips_j_elim_global, delimiter=",", fmt="%d")
        np.savetxt(d / "region_labels.csv", self.ips_region_labels, delimiter=",", fmt="%d")
        np.savetxt(d / "acceptances.csv", self.ips_acceptances, delimiter=",")
        np.savetxt(d / "num_samples.csv", self.ips_num_samples, delimiter=",")

        l_matrices_assoc = []
        for r, region_blocks in enumerate(self.ips_Ls, start=1):
            l_matrices_assoc.append({
                "region": r,
                "blocks": [
                    {"real": L.real.tolist(), "imag": L.imag.tolist()}
                    for L in region_blocks
                ],
            })
        l_json = {
            "num_regions": len(self.ips_Ls),
            "num_blocks_per_region": (len(self.ips_Ls[0]) if self.ips_Ls else 0),
            "block_sizes": [int(a + 1) for a in self.ambient],
            "L_matrices": l_matrices_assoc,
        }
        with open(d / "L_matrices.json", "w") as f:
            json.dump(l_json, f, indent=2)

        metadata = self._build_metadata_dict()
        with open(d / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Wrote raw IPS outputs to {d}")

    def _build_metadata_dict(self) -> dict:
        return {
            "schema_version": 4,
            "generator": "GeneratePointsMCICYIPS",
            "requested_total_points": int(self.ips_total_pts_requested),
            "num_regions_requested": int(self.ips_num_regions),
            "precision": int(self.precision),
            "verbose": int(self.verbose),
            "front_end": bool(self.ips_front_end),
            "kappa_sample_size": int(self.kappa_sample_size),
            "num_points_valid": int(len(self.ips_points)),
            "point_dimension": int(self.ncoords),
            "dim_cy": int(self.ips_dim_cy),
            "num_hypersurfaces": int(len(self.coefficients)),
            "num_ambient_factors": int(len(self.ambient)),
            "dimPs": self.ambient.tolist(),
            "ambient": self.ambient.tolist(),
            "exponents": [m.tolist() for m in self.monomials],
            "coefficients_realimag": [
                [{"re": float(c.real), "im": float(c.imag)} for c in eq.astype(np.complex128)]
                for eq in self.coefficients
            ],
            "kahler_moduli": np.asarray(self.kmoduli, dtype=np.float64).tolist(),
            "target_volume": (
                None if getattr(self, "vol_j_norm", None) is None
                else float(self.vol_j_norm)
            ),
            "omega_quantity": "Omega (complex)",
            "omega_description": (
                "The Mathematica sampler returns the complex holomorphic "
                "volume-form factor Omega = 1/det(B), matching the omega "
                "convention of the other point generators. omega_real.csv / "
                "omega_imag.csv store its real/imag parts; omega_squared.csv "
                "stores the derived |Omega|^2 = Re[Omega*conj(Omega)] (which "
                "is also what dataset.npz carries in the second y column)."
            ),
            "weights_quantity": (
                "acceptance_rate_region * (|Omega|^2 / top_form_det); "
                "regional kappa is stored separately in kappas.csv. "
                "weights.csv always holds these raw Mathematica weights; "
                "when normalize_to_vol_j is true, dataset.npz weights are "
                "additionally rescaled in Python so that "
                "int_X det(g) = kappa_ijk t^i t^j t^k."
            ),
            "weights_acceptance_corrected": True,
            # Records the flag actually in effect for this run (set by
            # prepare_dataset; falls back to the class default when metadata
            # is built outside prepare_dataset).
            "normalize_to_vol_j": bool(
                getattr(
                    self, "ips_normalize_to_vol_j",
                    self.DEFAULT_NORMALIZE_TO_VOL_J,
                )
            ),
            "patches_local_convention": (
                "1-indexed patch index within each projective block"
            ),
            "patches_global_convention": (
                "1-indexed flattened global coordinate indices"
            ),
            "j_elim_global_convention": (
                "1-indexed flattened global eliminated coordinate indices"
            ),
            "region_labels_convention": (
                "1-indexed metric/region label for each accepted point"
            ),
            "files": {
                "points_real_csv": "points_real.csv",
                "points_imag_csv": "points_imag.csv",
                "weights_csv": "weights.csv",
                "omega_real_csv": "omega_real.csv",
                "omega_imag_csv": "omega_imag.csv",
                "omega_squared_csv": "omega_squared.csv",
                "kappas_csv": "kappas.csv",
                "patches_local_csv": "patches_local.csv",
                "patches_global_csv": "patches_global.csv",
                "j_elim_global_csv": "j_elim_global.csv",
                "region_labels_csv": "region_labels.csv",
                "acceptances_csv": "acceptances.csv",
                "num_samples_csv": "num_samples.csv",
                "L_matrices_json": "L_matrices.json",
            },
        }

    def _write_cymetric_outputs(
        self, d: Path, val_split: float, shuffle: bool, seed: int,
        normalize_to_vol_j: bool = DEFAULT_NORMALIZE_TO_VOL_J,
        write_raw: bool = False,
    ) -> float:
        n = len(self.ips_points)
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        t_i = int((1.0 - val_split) * n)
        train_idx, val_idx = idx[:t_i], idx[t_i:]

        points = self.ips_points
        X = np.concatenate([points.real, points.imag], axis=1).astype(np.float64)
        # y second column is |Omega|^2, derived here from the complex Omega
        # exactly as cymetric's nphelper.prepare_dataset does; cymetric's loss
        # reads y[:, -1] as omega_squared.
        omega_squared = self.ips_abs_omega_squared()
        # Kaehler-volume normalization is applied to the exported weights
        # only; self.ips_weights keeps the raw Mathematica values so
        # weights.csv and det_eff = |Omega|^2 / w reconstruction stay intact.
        vol_j_norm_fac = (
            self.vol_j_normalization_factor() if normalize_to_vol_j else 1.0
        )
        weights_out = vol_j_norm_fac * self.ips_weights
        y = np.column_stack([weights_out, omega_squared]).astype(np.float64)

        logger.info(f"Computing val_pullbacks for {len(val_idx)} validation points...")
        val_pullbacks = self.pullbacks(points[val_idx])

        dataset_path = d / "dataset.npz"
        np.savez_compressed(
            dataset_path,
            X_train=X[train_idx], y_train=y[train_idx],
            X_val=X[val_idx], y_val=y[val_idx],
            val_pullbacks=val_pullbacks,
        )
        logger.info(f"Wrote {dataset_path}")

        # kappa = Vol_K / Vol_CY of the exported data, computed exactly as the
        # other generators compute it, so the basis is consistent with
        # dataset.npz by construction. The per-region ips_kappas belong to the
        # sampler's ownership test and live in its own normalization; they are
        # kept for the raw outputs only.
        kappa_for_basis = float(
            self.compute_kappa(points, weights_out, omega_squared)
        )
        self.prepare_basis(str(d), kappa=kappa_for_basis)
        logger.info(f"Wrote basis.pickle (kappa={kappa_for_basis:.6e})")

        if not write_raw:
            return kappa_for_basis

        # point_gen.pickle (cymetric pattern). Strip the wl session before pickling.
        old_session, self.wl_session = self.wl_session, None
        try:
            with open(d / "point_gen.pickle", "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Wrote {d/'point_gen.pickle'}")
        finally:
            self.wl_session = old_session

        order = np.concatenate([train_idx, val_idx])
        extras = {
            "patches_local_mathematica_1idx": self.ips_patches_local[order],
            "patches_local_python_0idx": self.ips_patches_local[order] - 1,
            "patches_global_mathematica_1idx": self.ips_patches_global[order],
            "patches_global_python_0idx": self.ips_patches_global[order] - 1,
            "j_elim_global_mathematica_1idx": self.ips_j_elim_global[order],
            "j_elim_global_python_0idx": self.ips_j_elim_global[order] - 1,
            "region_labels_mathematica_1idx": self.ips_region_labels[order],
            "region_labels_python_0idx": self.ips_region_labels[order] - 1,
            "acceptances": self.ips_acceptances,
            "num_samples": self.ips_num_samples,
        }
        if self.ips_Ls:
            n_regions_L = len(self.ips_Ls)
            n_blocks = len(self.ips_Ls[0])
            for r, region_Ls in enumerate(self.ips_Ls):
                for b, L in enumerate(region_Ls):
                    h = np.conj(L.T) @ L
                    extras[f"h_block_r{r}_b{b}_real"] = h.real.astype(np.float64)
                    extras[f"h_block_r{r}_b{b}_imag"] = h.imag.astype(np.float64)
            extras["h_blocks_num_regions"] = np.array([n_regions_L], dtype=np.int64)
            extras["h_blocks_num_blocks"] = np.array([n_blocks], dtype=np.int64)
        np.savez_compressed(d / "mathematica_extras.npz", **extras)
        logger.info(f"Wrote {d/'mathematica_extras.npz'}")

        summary = {
            "num_regions": int(self.ips_num_regions),
            "n_points_used": int(n),
            "n_coords": int(self.ncoords),
            "val_split": float(val_split),
            "shuffle": bool(shuffle),
            "seed": int(seed),
            "ambient": self.ambient.tolist(),
            "dimPs": self.ambient.tolist(),
            "vol_moduli": np.asarray(self.kmoduli, dtype=np.float64).tolist(),
            "kahler_moduli": np.asarray(self.kmoduli, dtype=np.float64).tolist(),
            "omega_quantity_from_metadata": "Omega (complex)",
            "dataset_y_columns": "[weight, |Omega|^2]",
            "normalize_to_vol_j": bool(normalize_to_vol_j),
            "vol_j_norm_factor": float(vol_j_norm_fac),
            "kappa_for_basis": float(kappa_for_basis),
            "dataset_file": "dataset.npz",
            "basis_file": "basis.pickle",
            "has_region_labels": True,
            "has_acceptances": True,
            "has_num_samples": True,
            "has_h_blocks": bool(self.ips_Ls),
        }
        with open(d / "pack_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote {d/'pack_summary.json'}")

        return kappa_for_basis
