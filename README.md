# Corrected Amortized Inference (CAI)

Code, finite-element datasets and numerical records accompanying:

**Jun Hu and Heung-fai Lam. A dual-mode amortized Bayesian framework for real-time structural model updating with MCMC correction. Engineering Structures, 2026, article 124118.**  
DOI: [10.1016/j.engstruct.2026.124118](https://doi.org/10.1016/j.engstruct.2026.124118)

CAI combines a differentiable structural surrogate with a conditional normalizing flow. The flow provides an approximate posterior for a new observation and can also transform coordinates for Hamiltonian Monte Carlo. An importance-weight overlap diagnostic selects fast inference or MCMC correction.

## What the framework provides

- **One flow, two uses.** Posterior approximation and flow-transported HMC share the same invertible map. This transport route is demonstrated on analytical and synthetic structural cases.
- **Observation-dependent correction.** Raw importance-sampling ESS divided by proposal count is the routing statistic. The two field targets use adaptive NUTS as their correction sampler.
- **Separate numerical error checks.** Posterior approximation is compared with MCMC references, and surrogate discrepancy is checked against independent FE evaluations.

Correction targets the surrogate-based posterior. For incomplete modal records, the observed frequencies define the likelihood while MAP-based completion supplies the flow's conditioning vector.

## Start with the released records

The following command uses only the Python 3.11+ standard library. It verifies the file manifest, the four FE datasets, all 50 Case C records and their stored likelihood-error summaries, and the MAP-completion timing summaries:

```bash
python scripts/verify_release.py
```

It reads existing results and performs no FE solves, training or MCMC. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the table-to-record map and [IMPLEMENTATION.md](IMPLEMENTATION.md) for model assignments and evaluation conventions.

## Repository contents

| Path | Contents |
|---|---|
| `01_generate_samples.py`–`06_merge.py` | FE data-generation and surrogate workflow |
| `experiments/cai_lib.py` | Shared surrogate, flow, likelihood and sampler classes |
| `experiments/corrected_amortized.py` | D = 6 conditional-flow and transport experiment |
| `experiments/benchmarks/` | Analytical benchmarks and plotting scripts |
| `experiments/coupled_building/` | Coupled-building model and training scripts |
| `d12_validation_v2.py` | Original D = 12 experiment |
| `apdl_template/` | ANSYS Mechanical APDL templates |
| `revision_experiments/` | Recorded experiment scripts, posterior arrays, diagnostics, timing records and FE comparisons |
| `scripts/` | Read-only release verification |
| `RELEASE_MANIFEST.json` | File sizes and SHA-256 checksums |

The dated experiment paths preserve the correspondence between scripts and saved records. The original drivers include machine-specific paths and output locations. Check those settings before using a driver for a new experiment; the read-only verification command above works directly from the repository root.

## Finite-element datasets

| File | Successful FE evaluations | Use |
|---|---:|---|
| `dataset_train.csv` | 8,009 | Sailboat training batch |
| `dataset_test.csv` | 1,995 | Independent Sailboat test batch |
| `dataset_train_broad_ext.csv` | 12,323 | Domain-extended Sailboat field analysis |
| `dataset_coupled_train.csv` | 5,000 | Coupled-building surrogate |

Training and test batches in the first two rows are independent. The 12,323-row dataset is the complete domain-extended dataset, not an additional batch to add to the 8,009 rows.

## Selected results and timing scope

For complete observations, the reported D = 6 fast-mode evaluation produces 2,000 proposal draws in about 7 ms. The six-mode Sailboat field record requires about 5.2 s for conditioning-vector completion and 19 ms for flow evaluation; its correction run takes 76.9 s. These are stage measurements with different output counts, not matched-effective-sample benchmarks. Both field cases route to correction for all five tested proposal seeds.

The coupled-building posterior-region FE comparison uses exactly 50 preselected draws from the independent Uniform(0.5, 4) three-coordinate posterior:

| Quantity | Case C |
|---|---:|
| Maximum first-six relative surrogate/FE error | 0.0343% |
| Maximum absolute log-likelihood difference | 0.0257 nat |
| Mean absolute log-likelihood difference | 0.00775 nat |

The complete records and numerical definitions are indexed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Environment and trained models

The research code uses Python, NumPy, SciPy, PyTorch, pandas, Matplotlib and Pyro; ArviZ supplies diagnostics. A typical installation is:

```bash
python -m pip install -r requirements.txt
```

ANSYS Mechanical APDL and a valid local license are needed only to generate new FE outputs. CUDA is optional for the record checks and useful for training.

Trained D = 6, domain-extended Sailboat and coupled-building models are available from the corresponding author on request. Original D = 12 weights were not saved. Its experiment script is included, but a new training run will not reproduce every historical digit. See the reproducibility record for checkpoint locations and other run-level availability.

## Citation

```bibtex
@article{hu2026cai,
  author  = {Hu, Jun and Lam, Heung-fai},
  title   = {A dual-mode amortized Bayesian framework for real-time structural model updating with MCMC correction},
  journal = {Engineering Structures},
  year    = {2026},
  pages   = {124118},
  doi     = {10.1016/j.engstruct.2026.124118}
}
```

Contact: Jun Hu, junhu22@whut.edu.cn. Code license: [MIT](LICENSE).
