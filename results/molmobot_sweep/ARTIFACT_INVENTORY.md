# Artifact inventory — what raw data backs each datapoint

Status of source artifacts behind every cell in the v4 poster results.
"raw" = committed per-trajectory results (and where available, per-episode logs)
that let the rate be recomputed from source. "aggregated-only" = the number
survives only in the chart PNGs and FINDINGS_AND_CAVEATS.md; the per-episode
files were cleaned from /tmp before they could be committed and are not
recoverable without re-running the eval.

## Headline (origpose hybrid, n=347/cell)
| module        | condition       | rate     | artifact |
|---------------|-----------------|----------|----------|
| MolmoBot      | static-hybrid   | 324/347  | aggregated-only |
| MolmoBot      | mobile@static   | 311/347  | aggregated-only |
| MolmoBot      | perturbed       | 280/347  | aggregated-only |
| pi0.5         | static-hybrid   | 72/347   | aggregated-only |
| pi0.5         | mobile@static   | 51/347   | aggregated-only |
| pi0.5         | perturbed       | 48/347   | aggregated-only |
| MolmoBot-Pi0  | perturbed       | 150/347  | RAW -> raw_runs/mbpi0_perturbed/ (results.jsonl + r2_gpu*.log.gz) |
| scripted IK   | all             | 347/347  | oracle, defines the scene set |

## Other committed result files (not the headline hybrid cells)
| file                   | rate    | note |
|------------------------|---------|------|
| static_results.jsonl   | 231/347 | MolmoBot static-STANDALONE harness (66.6%), different pipeline than the 324 hybrid cell |
| mobile_results.jsonl   | 135/347 | superseded "muddy v2" double-perturbed mobile run; kept for history |
| comparison.jsonl       | -       | static-vs-mobile pairing for the above |

## To close the gap
The six aggregated-only cells would need re-running to regenerate raw artifacts
(MolmoBot ~1-4 min/ep, pi0.5 ~11-13 min/ep; perturbation is reproducible from
seed=hash((0,house,episode))). Until then they are provable only at the
aggregate level via the charts + FINDINGS.
