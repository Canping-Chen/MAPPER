"""
Default configuration dataclasses for MAPPER (v0.7).

This module owns only the tunable knobs.  Methodology and the relationship
between the pieces are documented in :mod:`mapper.api`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TeacherConfig:
    """Bulk MLP — intrinsic phenotype program."""
    hidden_dim: int = 512
    embed_dim: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    batch_size_max: int = 64
    val_frac: float = 0.2
    # Two-stage refit policy.  ``refit_full`` is the manual override:
    #   None  : AUTO (default) — decide from cohort size / balance (see below)
    #   True  : always run stage-2 full-data refit
    #   False : never refit; deploy the stage-1 early-stopped model
    # AUTO rule: refit on ALL bulk samples (for best_epoch+1 epochs, no holdout)
    # when the cohort is small OR imbalanced — i.e. when ``n_bulk < refit_min_bulk``
    # OR the minority class/event fraction ``< refit_min_class_frac``.  Only when
    # n_bulk ≥ refit_min_bulk AND both sides ≥ refit_min_class_frac do we skip the
    # refit and deploy the stage-1 model (keeping a clean holdout val estimate).
    # Continuous mode has no class proportion, so only the size criterion applies.
    refit_full: Optional[bool] = None
    refit_min_bulk: int = 200
    refit_min_class_frac: float = 0.10


@dataclass
class GraphConfig:
    """Spatial radius graph (spatial mode only)."""
    max_neighbors: int = 30
    spatial_chunk_size: int = 50_000
    gpu_n_threshold: int = 50_000


@dataclass
class NicheConfig:
    """Niche scoring + likelihood-ratio grouping (spatial mode).

    A likelihood-ratio classifier compares per-spot scores against the full
    bulk training distribution, avoiding the scale mismatch between aggregated
    bulk and single-spot views.
    """
    # which spatial refinement to use
    #   "pseudo_bulk" : s_niche = teacher_MLP(LOO pseudo-bulk)  [default, interpretable]
    #   "gnn"         : s_niche = StudentGNN(teacher_emb, G_spatial), distilled from pseudo_bulk
    #   "none"        : ablation — no spatial refinement, no niche tier in grouping
    refine_method: str = "pseudo_bulk"
    # final composite score for visualisation: s_final = (1-α) s_cell + α s_niche
    # (this is only used for MAPPER.Score output; grouping uses LRs of s_cell
    # and s_niche separately)
    refine_alpha: float = 0.5
    # whether the local pseudo-bulk includes the central spot (LOO if False)
    include_self_in_pseudobulk: bool = False
    # likelihood-ratio threshold in natural-log units
    #   0.0 : binary (Bayes-optimal under uniform prior)
    #   0.5 : moderate evidence (Bayes factor ≳ 1.65)  ← default
    #   1.0 : strong evidence (BF ≳ 2.72)
    log_lr_threshold: float = 0.5
    # KDE bandwidth for the bulk distributions
    #   "scott" : Scott's rule (default, robust)
    #   "silverman" or float : alternative bandwidths
    kde_bandwidth: str = "scott"
    # cap |log_LR| for numerical stability in distribution tails
    log_lr_clip: float = 20.0
    # Negative-side threshold τ_neg is calibrated to a false-omission rate:
    # at most ``neg_foi_q`` of the bulk positives may fall into the negative
    # region {log_LR < -τ_neg}.  τ_neg deepens automatically as the bulk
    # positive / negative score distributions overlap, and is floored at τ.
    # (see grouping._calibrate_tau_neg)
    #
    # Adaptive escalation (same rule for scRNA / spatial, single / multi):
    # start at the strict ``neg_foi_q``; group; measure the negative fraction
    # (for scRNA *after* the GMM refinement; for multi-slide *pooled* over all
    # slides = Σ neg / Σ spots).  While that fraction < ``neg_frac_floor`` the
    # negative reference is too sparse, so climb the ascending ``neg_foi_q_steps``
    # ladder one rung at a time (0.01 → 0.025 → 0.05), stopping at the first rung
    # that meets the floor — or at the last rung if none do.  Then it is fixed,
    # regardless of the resulting fraction.  ``neg_foi_q`` (an FOI level on the
    # bulk positives) and ``neg_frac_floor`` (a fraction of the *target* spots)
    # are independent quantities.
    neg_foi_q: float = 0.01
    neg_foi_q_steps: tuple = (0.025, 0.05)
    neg_frac_floor: float = 0.05


@dataclass
class StudentConfig:
    """Optional Student GNN, used only when ``NicheConfig.refine_method == 'gnn'``.

    The GNN is trained to **distill the pseudo-bulk niche score** from the
    teacher embedding and the spatial topology:

        s_gnn = StudentGNN(teacher_emb, G_spatial)
        loss  = MSE(s_gnn, s_niche_pseudobulk) + lambda_smooth * Lap(s_gnn, G_spatial)

    This produces a smoother, parameter-efficient predictor that can be
    applied without re-computing pseudo-bulk at inference time.
    """
    hidden_dim: int = 128
    out_dim: int = 64
    dropout: float = 0.2
    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    lambda_smooth: float = 0.1
    neighbor_sizes: tuple = (20, 10)
    minibatch_size: int = 8192


@dataclass
class VAEConfig:
    hidden_dim: int = 128
    latent_dim: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    kl_weight: float = 1e-3


@dataclass
class NBVAEConfig:
    hidden_dim: int = 256
    latent_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    kl_weight: float = 1e-6
    batch_size: int = 1024


@dataclass
class GraphVAEConfig:
    hidden_dim: int = 256
    latent_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    kl_weight: float = 1e-7


@dataclass
class MapperConfig:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    niche: NicheConfig = field(default_factory=NicheConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    nbvae: NBVAEConfig = field(default_factory=NBVAEConfig)
    graphvae: GraphVAEConfig = field(default_factory=GraphVAEConfig)
    reconstruct: bool = False
    n_select_genes: int = 0
    gene_select_method: str = "elasticnet"
    teacher_inference_batch_size: int = 8192
    # ---- multi-slide normalisation strategy ----
    # Used by ``mapper_multi_slide`` only.  Controls how target expression is
    # z-scored before being fed to the (shared) teacher MLP:
    #
    #   "joint"          : concatenate all slides, then z-score per gene
    #                      (DEFAULT).  Cross-slide differences in cell-type
    #                      composition are preserved -- a slide that is mostly
    #                      normal tissue correctly gets a low pos%, a slide
    #                      that is mostly aggressive tumour correctly gets a
    #                      high pos%.  Empirically required for PAAD where
    #                      Pt-12A (mostly normal pancreas) should have ~0 %
    #                      positive spots and dense tumour slides should have
    #                      many.  Deep slides can be slightly biased toward
    #                      higher z-scores by gene-detection differences.
    #
    #   "per_slide"      : each slide z-scores itself.  No depth artefact but
    #                      cross-slide content differences are erased -- every
    #                      slide ends up with similar pos% regardless of true
    #                      tumour content.  Useful only for within-slide
    #                      relative ranking.
    multi_slide_normalization: str = "joint"


DEFAULT = MapperConfig()
