# Raw run artifacts (verifiable source data for the poster charts)

Each subdir holds the unaggregated per-trajectory results + driver logs for one
eval run, so the chart numbers can be reproduced/audited.

- `mbpi0_perturbed/` — MolmoBot-Pi0-DROID served as grasp module, perturbed
  condition, full 347 trajectories (origpose benchmark), 2026-06-12.
  `mbpi0_mobile_results.jsonl`: 150/347 = 43.2% (Wilson 95% [38.1, 48.5]).
  `r2_gpu{0-7}.log`: per-episode logs incl. `perturbed by (dx,dy)` for radius.

NOTE: the clean MolmoBot and pure pi0.5 perturbed raw per-episode files
(formerly /tmp/mb_*, /tmp/pi05_*) were cleaned from /tmp before this commit and
could not be recovered; their aggregated final numbers live in the charts_v4
FINDINGS_AND_CAVEATS.md and the committed *_results.jsonl.
