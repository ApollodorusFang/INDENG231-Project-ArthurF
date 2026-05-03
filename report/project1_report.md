**Arthur Fang** — INDENG 231 Project 1 — April 29, 2026

---

## 1. Backtesting System Design

**Data layer.** `src/data_loader.py` loads close-price data into a date
× ticker DataFrame; both wide and long CSV layouts are auto-detected.
Missing prices are forward-filled and tickers that remain entirely NaN
are dropped. Daily simple returns
$R_{i,t} = P_{i,t}/P_{i,t-1} - 1$ live in a parallel panel of identical
shape.

**Strategy contract.** `src/strategies/base.py` defines
`BaseStrategy.generate_weights(prices_until_t, returns_until_t,
current_date)`, returning a target weight Series for the current day.
Subclasses override that single method; the engine handles everything
else. A shared `validate_weights` method clips negatives to zero,
normalizes if the gross exceeds 1, and treats the residual as cash.
Three reusable weighting helpers (`equal_weight`,
`inverse_volatility_weight`, `rank_weight`) live in
`src/portfolio.py`.

**Engine.** `src/backtester.py` iterates the trading calendar. At day
*t* it slices prices/returns to rows `[: t]`, asks the strategy for
*w<sub>t</sub>*, validates, and *realizes those weights against
R<sub>i,t+1</sub>*:

$$
R_{p,\,t+1} = \sum_i w_{i,t} R_{i,t+1} - \kappa\,\lVert w_t - w_{t-1}\rVert_1,
\qquad
\mathrm{NAV}_{t+1} = \mathrm{NAV}_t (1 + R_{p,\,t+1}).
$$

The strategy never sees *R<sub>i,t+1</sub>* when choosing
*w<sub>i,t</sub>*; this is the structural no-lookahead guarantee.
An optional `update_after_return` hook lets adaptive strategies
(the UCB meta-strategy in §6) receive realized rewards, fired *after*
*R<sub>p,t+1</sub>* is computed.

**Metrics.** `src/metrics.py` computes cumulative return, annualized
return, annualized volatility, Sharpe (zero risk-free rate), maximum
drawdown, Calmar, win rate, and average L1 turnover. All
annualizations use *T = 252*.

**Outputs.** Each phase writes one CSV (`outputs/tables/`), two PNGs
(`outputs/figures/`, NAV + drawdown overlays), and one JSON run log
(`outputs/logs/`).

**Assumptions.** (i) Weights execute at the recorded close;
(ii) zero risk-free rate; (iii) long-only, no leverage;
(iv) transaction cost defaults to 0 (the engine accepts a
proportional *κ* for sensitivity); (v) no corporate-action handling,
prices treated as adjusted; (vi) static universe equal to the loaded
CSV (survivorship-biased).

**Module map.** `src/` (config, data_loader, backtester, metrics,
plotting, portfolio, risk), `src/strategies/` (base, single_stock,
benchmarks, cross_sectional, bandit), `experiments/` (one entry
script per deliverable), `outputs/`, and `report/`.

---

## 2. Design Choices

**Next-day return realization, not shifted weights.** The strategy
sees `prices[:t]` (close of *t* inclusive) and is paid by
*R<sub>i,t+1</sub>*. The alternative — pay *w<sub>t</sub>* with
*R<sub>i,t</sub>* — would force the strategy to choose with
information through *t−1*, which conflates "observe the close" with
"trade on the close". The chosen pattern matches how a real desk
works: decide at the close, realize the move overnight to the next
close.

**Constraint enforcement in the base class.** `validate_weights`
clips negatives, normalizes if gross > 1, and asserts the invariants.
Pushing this into the engine means a new strategy author cannot
accidentally short or lever, and the engine never has to trust the
subclass.

**Positive-return gate on the new momentum strategies.** Restricting
selection to names with *ρ<sub>i,t</sub>* > 0 converts top-*K*
momentum into a long-or-cash rule for free: in a broad-market sell-off
the entire cross-section can be negative and the strategy parks in
cash instead of loading on the "least bad" names.

**K=3 with inverse-volatility weighting.** A concentrated portfolio
is dominated by idiosyncratic vol, so inverse-vol weighting matters
more there than at *K = 10* where the spread is already diluted. The
grid agreed with this intuition — the inverse-vol variant won at
*K = 3* but only marginally at *K = 10*. Sitting at the boundary
(*K = 3*, *ℓ = 90*) is flagged in §5 as an overfitting risk; with
this sample length the in-sample optimum clearly wants *more*
concentration and *longer* memory than the grid offers, but *K = 1*
is no longer a portfolio strategy.

**Drawdown threshold −10% on a 120-day window.** −5% triggers on
routine intra-month volatility (too many false positives); −15% only
fires after the worst is already past (too late). 120 trading days
≈ 6 months gives a smooth medium-term drawdown signal that does not
whipsaw on single-week sell-offs. The grid confirmed −10% as the
in-sample winner.

**60/40 split rather than walk-forward.** A single split is
transparent and easy to audit, which is the right trade-off for a
course project. Walk-forward retuning would reduce overfitting more,
at the cost of an extra hyperparameter (refit frequency) and a much
heavier compute bill — worth doing as a follow-up but not the headline
protocol here.

---

## 3. Deliverable 3 — Single-Stock Strategy Comparison

Test five long-or-cash strategies on NVDA over the full 1,255-day
sample: 20-day momentum; 20-day mean reversion (long after a −5%
trailing return); 20/50 SMA crossover; volatility-filtered momentum
(long iff trailing return > 0 AND 20-day vol < 3%); and 20-day z-score
mean reversion (long iff *z<sub>t</sub>* < −1). Each strategy is a
single-stock specialization of the generic `BaseStrategy` — no engine
code changes.

| Strategy | Cum. Ret. | Ann. Ret. | Ann. Vol. | Sharpe | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Momentum (20) | 3.694 | 0.364 | 0.384 | **0.997** | −0.498 | 0.095 |
| Z-score mean reversion (20) | 2.022 | 0.249 | 0.262 | 0.977 | **−0.226** | 0.110 |
| Vol-filtered momentum | 2.131 | 0.258 | 0.272 | 0.974 | −0.364 | 0.076 |
| MA crossover (20/50) | 3.042 | 0.324 | 0.376 | 0.930 | −0.636 | **0.022** |
| Mean reversion (−5%) | 1.838 | 0.233 | 0.298 | 0.851 | −0.407 | 0.075 |

![Single-stock NAV curves on NVDA.](../outputs/figures/single_stock_nav.png)

![Single-stock drawdown profiles on NVDA.](../outputs/figures/single_stock_drawdown.png)

**Interpretation.** The top three Sharpes (momentum 0.997, z-score
0.977, vol-filtered 0.974) are within 0.02 of each other — well inside
the noise band of a 5-year single-asset Sharpe estimate (the
approximate SE at *N = 1255* is ≈ 0.45; see §6). Momentum captures the
most return (3.7×) but pays a −49.8% drawdown; z-score mean reversion
spends most of the period in cash and gives back roughly half the
return for less than half the drawdown, which is why it tops the
Calmar ranking. MA crossover is the lowest-turnover rule by an order
of magnitude.

---

## 4. Deliverable 4 — Portfolio-Level Backtest

Four constructions are tested over the full 101-stock pool. Two pin
the selection rule to the top-10 names by trailing 30-day return and
vary *only* the weighting scheme, isolating the uniform vs.
inverse-vol comparison required by the rubric. The SMA crossover
portfolio (equal-weighted) and the equal-weight buy-and-hold of the
entire pool are included as anchors. Inverse-volatility uses a 20-day
rolling daily std.

| Strategy | Wt. | Cum. Ret. | Ann. Vol. | Sharpe | Max DD | Turnover |
|---|:---:|---:|---:|---:|---:|---:|
| Top-10 momentum (lb=30) | Unif | 2.372 | 0.276 | **1.022** | −0.430 | 0.328 |
| Risk-adj top-10 momentum | IV | 1.187 | **0.181** | 0.961 | **−0.275** | 0.443 |
| SMA crossover (20/50) | Unif | 1.964 | 0.357 | 0.752 | −0.255 | **0.082** |
| Equal-weight buy-and-hold | Unif | 0.930 | 0.207 | 0.741 | −0.310 | 0.014 |

![Portfolio NAV. Uniform top-10 momentum dominates on cumulative growth.](../outputs/figures/benchmark_nav.png)

![Portfolio drawdowns. Inverse-vol weighting nearly halves the worst drawdown of top-10 momentum.](../outputs/figures/benchmark_drawdown.png)

**Interpretation.** On the same selection rule, uniform weighting
edges inverse-vol on Sharpe (1.02 vs. 0.96) and decisively wins on
cumulative return (2.37× vs. 1.19×); inverse-vol cuts volatility by
about a third (27.6% → 18.1%) and the drawdown by roughly half
(−43% → −27%). The reason uniform wins on Sharpe in this sample is
regime-specific: the highest-momentum names in 2021–2026 are also the
highest-volatility mega-cap tech names, so inverse-vol weighting
under-allocates to exactly the winners. This is the empirical
motivation for the Deliverable-5 strategies that *concentrate* into
the strongest momentum names rather than dilute them via vol scaling.

---

## 5. Deliverable 5 — Benchmarks vs. Two New Strategies

**Benchmark 1 (SMA crossover).** Equal-weight Nasdaq-100 stocks whose
20-day SMA exceeds the 50-day SMA; cash if none qualify.
`SMACrossoverPortfolioStrategy(20, 50)`.

**Benchmark 2 (Top-K momentum).** Equal-weight the top 10 names by
trailing 30-day return. `TopKMomentumStrategy(30, 10)`.

**New strategy 1 — ConcentratedMomentum.** Top-*K* by raw trailing
*ℓ*-day return, restricted to names with strictly positive trailing
return, weighted equal- or inverse-vol (chosen by the grid).
Concentrating into a small *K* raises the loading on the strongest
momentum signal; the positive-return gate parks the portfolio in cash
during broad-market downturns. `ConcentratedMomentumStrategy(lookback,
k, weighting, volatility_window)`.

**New strategy 2 — MomentumWithDrawdownControl.** Same top-*K*
positive-return selection, equal-weighted; if the equal-weight market
index has drawn down by more than |thr| over the last `market_window`
days, gross exposure is scaled to `defensive_exposure` (residual in
cash). The drawdown signal uses only past returns.
`MomentumWithDrawdownControlStrategy(lookback, k, drawdown_threshold,
defensive_exposure, market_window)`.

**Protocol.** Tune each new strategy on the first 60% of dates
(2021-04-13 → 2024-04-09) by maximizing in-sample Sharpe over a small
grid; evaluate the tuned strategies and the two benchmarks on the
held-out 40% test window (2024-04-10 → 2026-04-10, 502 days). Grid
for ConcentratedMomentum: `lookback ∈ {20,30,45,60,90}`,
`k ∈ {3,5,7,10,15}`, `weighting ∈ {equal, inverse_vol}`,
`vol_window=20`. Grid for MomentumWithDrawdownControl:
`lookback ∈ {20,30,45,60,90}`, `k ∈ {3,5,7,10}`,
`drawdown_threshold ∈ {−0.05, −0.10, −0.15}`,
`defensive_exposure=0.5`, `market_window=120`. Both grid winners
selected `k=3` and `ℓ=90` (Concentrated chose `inverse_vol`;
DD-control chose `thr=−0.10`).

| Strategy | Cum. Ret. | Ann. Ret. | Ann. Vol. | Sharpe | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| **ConcentratedMom** (k=3, ℓ=90, IV) | 4.103 | 1.266 | 0.477 | **1.951** | −0.397 | 0.236 |
| **MomDDCtrl** (k=3, ℓ=90, thr=−10%) | 3.843 | 1.207 | 0.465 | **1.937** | −0.392 | 0.173 |
| B2: Top-10 momentum (lb=30) | 1.246 | 0.501 | 0.279 | 1.594 | −0.288 | 0.302 |
| B1: SMA crossover (20/50) | 0.306 | 0.143 | 0.167 | 0.884 | **−0.181** | **0.069** |

![Test-window NAV. Both new strategies finish well above the top-K benchmark.](../outputs/figures/benchmarks_vs_new_nav.png)

![Test-window drawdowns. The new strategies pay for upside with deeper drawdowns (≈ −40%) than top-K (−29%).](../outputs/figures/benchmarks_vs_new_drawdown.png)

**Result.** Both new strategies beat both benchmarks on Sharpe.
ConcentratedMomentum at 1.951 and MomDDCtrl at 1.937 clear the top-K
benchmark (1.594) by 0.34–0.36 and the SMA benchmark (0.884) by more
than a full unit. The verdict is recorded in
`outputs/logs/new_strategies_log.json::comparison_vs_benchmarks` with
`beats_both_benchmarks: true` for both strategies.

**Caveats.** Three caveats temper the read. (i) The Sharpe gain comes
mostly from amplified *return* (≈ 2.5× top-K's annualized return) and
only partly from a worse *volatility* trade (≈ 1.7×); MDD is
correspondingly deeper (−40% vs. −29%). Higher Sharpe is not lower
risk. (ii) Both grid winners sit at the boundary of the search
(`k=3` smallest, `ℓ=90` longest), a structural overfitting flag — the
true optimum may lie outside the grid. (iii) On the test window the
standard error of an annualized Sharpe of 1.95 with *N = 502* is
$\sqrt{(1+\tfrac{1}{2}\widehat{\mathrm{SR}}^2)/N}\cdot\sqrt{T}\approx 0.71$,
so the 95% CI on the new strategies overlaps with top-K's CI; beating
the top-K benchmark by 0.34 Sharpe is real evidence but not
statistically overwhelming. The strategies should be re-validated under
walk-forward retuning, realistic transaction costs, and market regimes
other than the 2024–2026 AI rally before being treated as
production-ready.

---

## 6. Extensions: UCB Meta-Strategy and Risk Diagnostics

**UCB-1 over four arms.** `UCBMetaStrategy(arms, c=1.0)` selects each
day by $\mathrm{UCB}_k(t) = \overline{r}_k + c\sqrt{\ln T(t)/n_k(t)}$
across SMA crossover, top-10 momentum, risk-adjusted momentum
(ℓ=60), and low-volatility momentum (ℓ=60); the chosen arm's weights
are forwarded and `update_after_return` credits that arm's reward
after the realized *R<sub>p,t+1</sub>*. Selection counts on the full
sample are nearly uniform (315/315/313/311 of 1,254 trials) because the
per-day reward differences are sub-basis-point and the exploration
bonus dominates.

| Strategy | Cum. Ret. | Ann. Vol. | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|
| Top-10 momentum (lb=30) | 2.372 | 0.276 | **1.022** | −0.430 | 0.642 |
| Risk-adj momentum (ℓ=60) | 0.998 | 0.169 | 0.908 | **−0.200** | 0.747 |
| **UCB meta** (c=1, n=4) | 2.079 | 0.361 | 0.768 | −0.234 | **1.084** |
| SMA crossover (20/50) | 1.964 | 0.357 | 0.752 | −0.255 | 0.957 |
| Low-vol momentum (ℓ=60) | 0.399 | 0.149 | 0.528 | −0.215 | 0.325 |

UCB does not improve Sharpe over the best individual arm but cuts max
drawdown by ~20pp vs. top-K and posts the best Calmar in the table.
The cost is ~4× higher turnover; under any realistic transaction cost
the Sharpe gap would widen against UCB.

**Tail risk and Sharpe uncertainty.** `src/risk.py` computes empirical
5%-VaR / 5%-CVaR, an i.i.d. bootstrap of NAV paths, and an approximate
normal CI for the annualized Sharpe under i.i.d.-Gaussian returns,
$\mathrm{SE}(\widehat{\mathrm{SR}}) \approx
\sqrt{(1 + \tfrac{1}{2}\widehat{\mathrm{SR}}^2)/N}\cdot\sqrt{T}$. This
is not a full Lo (2002) HAC adjustment; the intervals widen if returns
are autocorrelated or fat-tailed. Top-10 momentum has the worst tail
(CVaR −3.82%, MDD −43%); risk-adjusted momentum (ℓ=60) has the
cleanest tail (CVaR −2.51%, MDD −20%). On the full sample only top-10
momentum's 95% Sharpe CI ([0.14, 1.90]) sits strictly above zero; the
other arms' CIs all include zero, so on this sample length differences
smaller than ~0.5 Sharpe are statistically indistinguishable from
noise.

![Bootstrap NAV paths for top-10 momentum. With identical mean and variance, the 5th–95th percentile spread of terminal NAV is roughly 8× (1.18 → 9.18); much of any realized track record is sequence luck.](../outputs/figures/bootstrap_nav_paths.png)

---

## 7. Limitations and Reproducibility

**Limitations.** Daily-close fillability with no spread or slippage;
zero default transaction cost (high-turnover strategies — UCB at 1.40,
the new k=3 momentum at 0.17–0.24, top-K at 0.30 — would lose
materially more alpha than the SMA portfolio at 0.07 under realistic
costs); survivorship bias in the constituent panel; single 60/40 split
with boundary-parameter selection in Deliverable 5; a single dominant
macro regime (post-COVID rates → AI mega-cap rally) in the 5-year
window; Sharpe CIs assume i.i.d. returns.

**Reproducibility.** Place the dataset at
`data/nasdaq100_daily_5y.csv` and run from the project root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/run_all.py            # engine smoke test
python experiments/run_single_stock.py   # D3
python experiments/run_benchmarks.py     # D4
python experiments/run_new_strategies.py # D5
python experiments/run_ucb_strategy.py   # extension
python experiments/run_risk_analysis.py  # extension
```

(On macOS, set `MPLBACKEND=Agg` before running plot-producing scripts
to avoid backend autodetection.) Each script writes a CSV metrics
table, NAV/drawdown PNGs, and a JSON log to `outputs/tables/`,
`outputs/figures/`, `outputs/logs/`. The Deliverable-5 verdict is
computed in `outputs/logs/new_strategies_log.json`.
