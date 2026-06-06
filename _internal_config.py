"""
Default configuration dataclasses for MAPPER (v0.7).

This module owns only the tunable knobs.  Methodology and the relationship
between the pieces are documented in :mod:`mapper.api`.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TeacherConfig:
    """Bulk MLP — intrinsic phenotype program."""
    hidden_dim: int = 256
    embed_dim: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    batch_size_max: int = 64
    val_frac: float = 0.2


@dataclass
class GraphConfig:
    """Spatial radius graph (spatial mode only)."""
    max_neighbors: int = 30
    spatial_chunk_size: int = 50_000
    gpu_n_threshold: int = 50_000


@dataclass
class NicheConfig:
    """Niche scoring + likelihood-ratio grouping (spatial mode).

    v0.6 — replaces the quantile-threshold design with a likelihood-ratio
    classifier that compares per-spot scores against the full bulk training
    distribution.  Removes the scale mismatch between aggregated bulk and
    single-spot views.
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
    # Negative-side rule: AND with data-adaptive deepening factor k
    # (see grouping.NEG_STRICT_CAP and note 30).  No knobs — k is
    # derived entirely from bulk class sizes.


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
class GroupingConfig:
    """scRNA Leiden cluster parameters used by ``cluster_phenotype_assign_lr``.

    The default spatial pipeline uses likelihood-ratio thresholds from
    ``NicheConfig.log_lr_threshold`` and does not read this config.
    """
    # expression PCA dimensionality for clustering
    pca_dim: int = 50
    # Leiden parameters
    cluster_n_neighbors: int = 15
    cluster_resolution: float = 1.0
    # Cluster-assignment rule: take top-N clusters by mean log_LR_cell and
    # additionally require each cell's own log_LR_cell to exceed the
    # NicheConfig.log_lr_threshold.  Top-2 was empirically validated to give
    # ~99 % cell-type purity on PRAD / HCC while keeping high recall.
    top_n_clusters: int = 2


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
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
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
