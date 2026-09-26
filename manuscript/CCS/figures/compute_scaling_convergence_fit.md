# Panel (a) convergence-onset fit

## Selection rule

Fit the continuous hinge model `y = b0 + m1*x + delta*max(0, x-t)` in log10(step)-log10(Test MAE) space. Optimize continuous breakpoint `t` for minimum total log10 residual SSE, require at least 3 formal checkpoints on each side, and require `abs(post_slope) <= 0.50 * abs(pre_slope)`. The 50% criterion was fixed before fitting and was not tuned against the aggregate R2. If it is infeasible, mark the curve weak/ambiguous and retain its best supported flattening point without deleting the curve.

## Selected points

| N | continuous_breakpoint_step | breakpoint_Test_MAE | pre_break_slope | post_break_slope | fit_residual_log10_sse | status |
|---:|---:|---:|---:|---:|---:|:---|
| 500 | 2864.938332 | 19.572747889384 | -0.674458 | -0.012922 | 0.204347393 | clear |
| 2500 | 3725.552487 | 16.716617589263 | -0.676772 | -0.016000 | 0.183768164 | clear |
| 10000 | 5788.053187 | 14.814347849709 | -0.640230 | -0.038283 | 0.169051445 | clear |
| 50000 | 3398.286289 | 18.039848504111 | -0.671439 | -0.131934 | 0.175359559 | clear |
| 100000 | 10000.000000 | 13.523545322480 | -0.599522 | -0.044146 | 0.208387682 | clear (support-boundary) |
| 280000 | 10000.000000 | 13.755364268447 | -0.597875 | -0.169739 | 0.198001851 | clear (support-boundary) |
| 465356 | 10000.000000 | 17.122624494650 | -0.559580 | -0.330572 | 0.153181632 | weak/ambiguous (support-boundary) |

## Power-law fit

Ordinary least squares in log10 space, matching Panels (b)/(c):

- Formula: `L = 86.153990687301 * C^(-0.193832538887)`
- a = `86.153990687301`
- k = `0.193832538887`
- R2 (log10 loss) = `0.593084663152`
- Number of representative points = `7`
- OLD discrete-breakpoint R2 = `0.5846`
- NEW continuous-breakpoint R2 = `0.593084663152`

The same seven curves are retained. Any R2 change comes from continuous breakpoint estimation and hinge-model interpolation, not from data filtering or modification.

No model was retrained and no existing experimental data or Panel (b)/(c) fit result was modified.
