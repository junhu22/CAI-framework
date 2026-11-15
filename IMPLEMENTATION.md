# Model and evaluation conventions

## Sailboat model

The ANSYS deck uses SHELL181 and BEAM189. A circular-solid beam section is a beam-section geometry, not a solid element type.

| Coordinate | Material use |
|---|---|
| e1 | Material 1: lower-storey/ground and secondary/front columns, plus roof/core shell layers |
| e2 | Material 2: higher-storey columns |
| e3 | Material 3: beam groups |
| e4–e6 | Materials 6–8: modulus definitions without active element assignments in this deck |

Shell sections 10 and 14 select material 1 in their layer data (`SECDATA,0.4,1` and `SECDATA,0.8,1`). Therefore an e1 perturbation changes its column and shell-layer groups together. The e1 AUC and recovery errors describe that composite material scale.

## Coupled-building model

The three-coordinate interface uses e1 for material 1, e2 for material 2 and e3 for material 3. Both shell sections (9 and 10) select material 1 in their layer data (`SECDATA,0.4,1` and `SECDATA,0.15,1`). Consequently e1 jointly scales column and shell-layer moduli and e3 scales beam moduli. The nominal e2 interface coordinate is not interpreted as an independently identified shell-stiffness parameter.

Normalized first-six surrogate sensitivity norms are (1.000, 0.00117, 0.860). These describe the learned implementation and are consistent with the layer-material selection; they are not a measurement of a separately varied slab stiffness.

## Partial observations and correction

Only observed frequencies enter the operational likelihood. For six-mode field records, MAP optimization on those modes supplies surrogate predictions for conditioning entries 7–20. Importance weights use the observed-data target in the numerator and the completed-vector proposal density in the denominator.

The original D = 6 likelihood uses kappa squared = 0.001. The coupled field and D = 12 configurations use kappa = 0.03. Raw IS-ESS/N is the operative router with threshold 0.5. PSIS ESS and Pareto-tail diagnostics are reported alongside it.

## E4 coverage and rank protocol

The original D = 6 flow is trained with untruncated LogNormal(0, 0.30 squared) draws. The E4 evaluation clips such draws componentwise to [0.5, 2.0], placing probability mass at the bounds.

The E4 observation generator is `y = f_hat(theta) * (1 + kappa * xi)`. The operational likelihood uses residuals `(f_hat(theta) - y) / y`. These are different noise conventions: the generator-implied likelihood would instead use a parameter-dependent scale and normalization.

Coverage and rank values, including the ten-observation correction spot check, summarize the recorded E4 evaluation protocol. The operational target, importance weights and correction target follow the article's relative-error likelihood.
