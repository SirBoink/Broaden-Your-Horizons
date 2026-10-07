# Results and interpretation

The saved transfer experiments show improved sequence completion with predictive motor subgoals. Supporting analyses examine robustness across starting conditions, base policies and numerical settings.

| Analysis | Saved comparison | Interpretation |
| --- | --- | --- |
| Historical transfer | Base 153/240; single discount 169/240; learned A–E 204/240 | Improved completion under the original 10 ms Euler protocol |
| Short-transition subgroup | Base 38/80; single 48/80; learned 74/80 | Encouraging performance in an exploratory subset of eight orders |
| Evaluation seeds | Base 154/240; DDQN 208/240 | Transfer benefit across starting conditions in a separate panel |
| Three base policies | Gains +22.50, +15.42 and +0.42 percentage points | Positive gains of varying size with a shared reacher and selector |
| Fine whole-sequence transfer | Base 17/240; DDQN 76/240; fixed A–E 59/240; single 65/240 | Transfer benefit in a finer simulator, with sensitivity to integration settings |
| Local replay at 0.5 ms | 29 improved, 38 tied and 13 reduced-progress starts among 80 | Complementary evidence about short-horizon handoffs |

These panels characterize the existing trained hierarchy. Whole-order bootstrap intervals and the short-transition subset are exploratory. Endpoint summaries average repeated invocations within episodes; trained policies, starting contexts, orders and human participants represent distinct units of replication. Additional selector fits and state-intervention records support further study of comparative library performance and physiological mechanisms.

Human passage profiles compare normalized prediction-scale shape. Curvature follows the published numerical rule with participant and order means. Regional reconstruction matches 2384 of 2400 published human values within 1e-6 m^-1; remaining differences are recorded in `data/human/curvature/results.json`. Published human summaries are preserved. Differences in task geometry and coverage guide the interpretation of these exploratory comparisons. The curvature panel on slide 36 links directly to its supporting records.

`presentation/slides.json` links the exact embedded assets to data and scripts. `data/provenance.json` records original source hashes; `data/manifest.json` records delivered files after path relocation. Frozen binary models and raw arrays retain their original bytes. Historical metadata preserves source context; execution resolves files inside this repository.
