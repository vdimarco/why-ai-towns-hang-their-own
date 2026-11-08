# Why AI Towns Hang Their Own

Code for the Swarm Alignment Hackathon report *Why AI Towns Hang Their Own: Herding and Model Lineage in
Multi-Agent Social Deduction* (Vaughn DiMarco, November 2026). The report is in this folder:
[`Why_AI_Towns_Hang_Their_Own.pdf`](Why_AI_Towns_Hang_Their_Own.pdf).

We reanalyze the belief probes from all 538 public games of
[Clocktower Bench](https://clocktowerbench.com), where AI players play Blood on the Clocktower. Each day, every
good player privately gives a probability that each other player is the Demon. We ask how independent these
forecasts are, when the town's group decision fails, and whether copies of one model think alike.

## Main results

- About 8 good players carry the information of 1.8 to 2.3 independent forecasters on day 1 and about 1.1 by
  day 5. A simulation control shows the measure scores independent agents as independent (7.1 to 7.2 of 8).
- Executions of the town's top suspect are wrong 1% of the time at 90%+ agreement, and 63% of the time below it.
  Good players nominated 68% of the wrong executions.
- In mixed Claude + GPT teams, same-model pairs agree more than cross-model pairs (+0.082 correlation, positive
  in 91% of 66 games), after controlling for private whispers.

## Audit

An AI-assisted audit of an earlier draft raised six concerns: agreement on the right answer, whether the measure
scores independent agents as one, related observations counted as separate, ties, novelty, and claims that change
with the corrected measure. Appendix B of the report lists each concern, the check and the outcome.

## Reproduce

```sh
pip install -r requirements.txt
./run_all.sh            # downloads the public data to ./data, writes every table and figure to ./results
```

It takes about five minutes and needs no API keys. Each step prints the numbers used in the report:

| Script | Report section |
|---|---|
| `analysis/00_download.sh` | Data (Section 3) |
| `analysis/01_probes.py` | Per-day and per-player tables, calibration |
| `analysis/02_independence.py` | Effective number and simulation control (Table 0, Section 4.1) |
| `analysis/03_executions_lineage.py` | Executions, nominations, lineage (Sections 4.2, 4.3) |
| `analysis/04_stats.py` | Confidence intervals and tests |
| `analysis/05_aggregation.py` | Aggregation methods (Section 4.4) |
| `analysis/06_figures.py` | Figures 1 to 3 |
| `analysis/07_robustness.py` | Game-level bootstrap and tie-aware checks (Appendix B) |

## What is pre-existing

- **Pre-existing, third-party:** the Clocktower Bench game engine, the 538 games and their belief probes, by
  Dwarkesh Patel ([github.com/dwarkeshsp/clocktower](https://github.com/dwarkeshsp/clocktower), MIT license).
  We ran no games. The data is downloaded from clocktowerbench.com and is not stored in this repository.
- **Written during the hackathon (November 7 to 8, 2026):** all code and the report in this repository.

The analysis code was written with help from Claude (Anthropic). See the LLM Usage Statement in the report.

## License

MIT. Blood on the Clocktower and its characters belong to The Pandemonium Institute.
