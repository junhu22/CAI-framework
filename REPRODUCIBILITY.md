# Reproducibility record

This record uses the table and figure numbering of the accepted Engineering Structures article, DOI 10.1016/j.engstruct.2026.124118. Numerical files are copied from the completed experiments. Their original timestamps and filenames identify the corresponding runs; they do not define the current table numbering.

## Read-only verification

Run `python scripts/verify_release.py` from this directory. The command checks file hashes and dataset counts, confirms 50 valid Case C frequency records, checks per-point likelihood arithmetic and reproduces summary reductions and timing medians from stored records. Surrogate predictions already reduced to per-point errors are read from the original metric CSV; evaluating the surrogate again requires the trained checkpoint.

## Table-to-record map

| Published result | Record or implementation |
|---|---|
| Tables 1–4, analytical benchmarks | `experiments/benchmarks/`; the summary plotting script contains the reported numerical literals |
| Table 5, material-coordinate definitions | `IMPLEMENTATION.md`, `apdl_template/`, and the successful Sailboat six-point input records |
| Tables 6–8, original D = 6 comparisons | Experiment implementations are included; original fast-mode, fixed-step HMC and transported-HMC run-level records were not retained |
| Table 6, tuned NUTS | `revision_experiments/E2_nuts_baseline/E2_results.json`, key `D6_Obs1` |
| Tables 9–10, three-seed ablation | [Ablation records](revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227/ablation_multiseed/), particularly `FULL_ABLATION_SUMMARY_RECOVERED.csv` and `PHASE_B_RESULTS.csv` |
| Table 11, routing and posterior gaps | [Four-chain reference records](revision_experiments/cai_r2_stage1/reference_gap_20260918_121500/), [boundary reference](revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/BOUNDARY/), [Sailboat completion](revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/SAILBOAT_PADDING/), [coupled reference](revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM/), and [width ratios](revision_experiments/cai_r2_provenance_closure_20260920_083412/ROUTING_GAP_WIDTH_RATIO.csv) |
| Table 12, threshold sensitivity | `revision_experiments/task_r3_domain/eth_final_results.json`, interpreted with the final field and near-threshold routing records |
| Table 13, coverage and rank diagnostics | `revision_experiments/E4_calibration_sbc/E4_results.json`; protocol details below |
| Table 14, D = 12 | Tuned-NUTS records in `E2_nuts_baseline/E2_results.json`; original per-scenario fast/transported-HMC records and D = 12 trained weights were not retained |
| Tables 15–16, coupled field target | [Coupled Uniform records](revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM/); use `COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz` and `COUPLED_NUTS_2000_GATE.json` |
| Tables 17–18, Sailboat field correction | `task_r3_domain/field_freeze_results.json` for the NUTS performance result; `nuts_ar_recover_results.json` for realized acceptance; [MAP-conditioned routing](revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/SAILBOAT_PADDING/) for final fast diagnostics |
| Table 19, field and perturbation routing | Final field-completion records above and [qualified boundary records](revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/BOUNDARY/) |
| Table 20 and Figure 17, group-resolved damage results | [Noise-consistent damage records](revision_experiments/cai_r2_stage1/r2_2_bounded_validation_noise_v2_20260919_074039/), including `DAMAGE_GRID_ACTIVE_GROUPS_V2.csv`, `DAMAGE_GRID_GROUP_RESOLVED_V2.csv` and `DAMAGE_GRID_TABLE18_DATA_V2.csv` |
| Independent FE holdouts, 216 matched inference pairs | [Matched arrays and summaries](revision_experiments/cai_r2_stage1/r2_2_bounded_validation_noise_v2_20260919_074039/), with the six successful inputs/frequencies in `cai_r2_stage1/local_fe_rerun_20260918_085458/` |
| Table 21, framework comparison | Qualitative architecture comparison; no additional experiment |
| Table 22, computational cost | `timing_results.json`, NUTS records above, and [MAP-completion timing](revision_experiments/cai_r2_provenance_closure_20260920_083412/MAP_COMPLETION_TIMING.csv) |
| Table A.23, Case A | `E1_posterior_region_FE/fe_caseA.csv`, `samples_caseA.csv`, `points_metadata.json`, and `task1_retrain/appendixA_numbers.json` |
| Table A.23, Case B | `task_r3_domain/fe_caseB_narrow.csv`, `samples_caseB_narrow.csv`, `points_metadata_narrow.json` and `caseB_narrow_stats.json` |
| Table A.23, Case C | [50-point HOST outputs](revision_experiments/cai_r2_caseC_HOST_RUN_20260920_125550/outputs/) and [HOST audit records](revision_experiments/cai_r2_caseC_HOST_DEBUG_20260920_133335/) |
| Table B.24 | Model/training settings in the included experiment sources |

Paths without a leading directory in the table refer to `revision_experiments/`. Some files retain earlier table numbers in their names. The mapping above identifies their use in the published article.

The Sailboat `field_freeze_results.json` also contains older proposal/transport results. Only its target-invariant NUTS performance result is used in the final field tables. Final proposal diagnostics use six observed frequencies and the MAP-completed conditioning vector. Likewise, the 2,000-draw-per-chain coupled reference supersedes shorter diagnostic chains.

## Case C: fixed selection and metrics

- Target: three independent Uniform(0.5, 4) coordinates.
- Observation: [2.06, 2.30, 2.62, 6.14, 6.53, 6.76] Hz.
- Likelihood: first six modes; kappa = 0.03.
- Reference: four chains with 2,000 retained draws each.
- Selection: PCG64 seed 20260920, 13/13/12/12 points from chains 811/812/813/814, without replacement and in saved generator order.
- Each selected point has one successful formal solve and 20 positive modal frequencies. Earlier license-failed launches did not produce scientific outputs.

| Quantity | Stored full-precision result |
|---|---:|
| delta_post | 0.0003430632202042094 |
| rho | 0.12004512595419847 |
| Standard bound | 2.287480442604919 nat |
| Residual-aware bound | 0.27494609110797125 nat |
| Maximum absolute log-likelihood difference | 0.02567717841688122 nat |
| Mean absolute log-likelihood difference | 0.007751269317860494 nat |
| All-20 maximum relative surrogate/FE error | 0.0010838951616620957 |

The original definition uses relative surrogate error `abs(prediction - FE) / FE`. Likelihood residuals and rho use the observed frequency in the denominator. The standard and residual-aware expressions are `M*delta*(2+delta)/(2*kappa^2)` and `M*delta*(2*rho+delta)/(2*kappa^2)`, respectively.

| Scientific input | SHA-256 |
|---|---|
| Four-chain reference | `c3174b9f49aed3cf0cd2ef68afddd7290ad10f47f5c20744ea60491cc1819d60` |
| Selected indices | `4188aca464376a1a4679206653f0281c686cbdff7c5e02d84637aca81a10817f` |
| Selected theta | `054bbbcbe4c75a604670faa5d6a63f9e8ec309c0a1dfdd853a968cfc2b427bf5` |
| Coupled surrogate checkpoint, available on request | `6a1e4f530d4e3628e0b7fdbf08661b4dc32a71a0c2b538b26f5af9d7c07b9986` |

Locked APDL inputs and per-point modal outputs are included. Solver databases and full host/license logs are excluded from the software release; their hashes remain in the original execution records. The historical license-state cause was not captured. The original HOST manifest predates an invocation-only script binding, which is documented in the later HOST integrity record. It should not be interpreted as a checksum of a later configured runner.

## Model files and historical limits

Available on request from Jun Hu (junhu22@whut.edu.cn):

| Model | Checkpoint location expected by the scripts |
|---|---|
| Original D = 6 surrogate | `surrogate_models/member_*.pt` |
| Original D = 6 flow | `experiments/results/conditional_flow.pt` |
| Domain-extended field surrogate and flow | `surrogate_models_broad_ext/member_*.pt`; `experiments/results/conditional_flow_broad_ext.pt` |
| Coupled surrogate and bounded-Uniform flow | `surrogate_models_coupled/member_0.pt`; `revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM/conditional_flow_coupled_uniform_primary.pt` |

The original D = 12 trained weights and original scenario-level run records were not saved. Training and observation generation in that experiment were not seeded end to end. The included script supports a new experiment, rather than exact recovery of historical digits.

Original plotting scripts for Figures 6, 7 and 11, and for an earlier cross-case summary figure, were not retained. The earlier summary figure is absent from the final article. Current repeated-ablation and damage data are supplied as numerical records. The analytical summary plotting script uses recorded literals, which are a summary rather than independent run logs.

The 19.0 ms Sailboat flow-evaluation timing has no original run-level timing record. The 5.2 s completion measurement has five recorded timed runs after one warm-up. The 76.9 s NUTS figure describes one performance chain with 500 warm-up and 500 retained steps; it is distinct from the four-chain posterior-gap reference. Coupled NUTS's approximately 352 s is a four-chain serial total.

## Coverage/rank evaluation and material definitions

[IMPLEMENTATION.md](IMPLEMENTATION.md) gives the exact material-coordinate mapping and the E4 evaluation convention. E4 uses a clipped simulation protocol and a generator whose noise scaling differs from the operational likelihood. Its coverage/rank summaries are descriptive for that protocol, not a matched-prior, matched-likelihood calibration guarantee for the operational target.

## Data lineage and integrity

The original 8,009-row training batch and 1,995-row test batch have no overlapping parameter vectors. The extended Sailboat set contains 12,323 rows. The domain extension adds 673 new FE analyses and replaces 373 earlier vectors, for a net increase of 300 relative to the prior 12,023-row set.

`RELEASE_MANIFEST.json` records the bytes and SHA-256 of each released file. Scientific datasets, posterior arrays, selections and FE outputs are copied without numerical changes. Original experiment scripts are included for traceability; new release utilities only read existing records.

Four module docstrings were updated for the published parameter definitions and inference scope. Their executable syntax trees are unchanged. New release utilities only verify stored records.
