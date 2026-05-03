# Figures to Include in the Final PDF

Each entry below lists the source PNG (relative to the repository
root), the report section in which it should be embedded, and a
one-sentence figure caption suitable for the PDF.

---

## Section 4 — Single-Stock Strategy Evaluation

1. **`outputs/figures/single_stock_nav.png`**
   *Figure 1.* Net asset value (NAV) curves of the five long-or-cash
   single-stock strategies on NVDA over 2021-04-13 → 2026-04-10,
   normalized to an initial capital of 1.0.

2. **`outputs/figures/single_stock_drawdown.png`**
   *Figure 2.* Drawdown curves
   (NAV<sub>t</sub> / max<sub>s≤t</sub> NAV<sub>s</sub> − 1)
   of the same five single-stock strategies on NVDA, highlighting the
   drawdown advantage of the mean-reversion variants.

---

## Section 5 — Portfolio Benchmark Strategies

3. **`outputs/figures/benchmark_nav.png`**
   *Figure 3.* NAV curves of the three Phase 3 benchmark portfolio
   strategies (SMA crossover, top-10 momentum, risk-adjusted top-10
   momentum) on the full 101-stock Nasdaq-100 universe.

4. **`outputs/figures/benchmark_drawdown.png`**
   *Figure 4.* Drawdown curves for the three benchmark portfolio
   strategies, illustrating that risk-adjusted top-K momentum has the
   shallowest drawdown despite its lower cumulative return.

---

## Section 6 — New Strategies vs. Benchmarks

5. **`outputs/figures/benchmarks_vs_new_nav.png`**
   *Figure 5.* Held-out test-window (2024-04-10 → 2026-04-10) NAV
   curves for the two new portfolio strategies (risk-adjusted momentum,
   low-volatility momentum) overlaid with the two benchmark strategies
   (top-10 momentum, SMA crossover).

6. **`outputs/figures/benchmarks_vs_new_drawdown.png`**
   *Figure 6.* Test-window drawdown curves for the same four
   strategies, showing that the new risk-adjusted strategies cut
   drawdown depth at the cost of trailing top-10 momentum on
   cumulative return.

---

## Section 7 — UCB Meta-Strategy

7. **`outputs/figures/ucb_nav.png`**
   *Figure 7.* NAV curves for the four base portfolio arms
   (SMA crossover, top-10 momentum, risk-adjusted momentum,
   low-volatility momentum) and the UCB-1 meta-strategy
   (`c=1`, `n=4`) over the full sample window.

8. **`outputs/figures/ucb_drawdown.png`**
   *Figure 8.* Drawdown curves for the four arms and the UCB
   meta-strategy, showing that arm rotation reduces the meta-strategy's
   maximum drawdown relative to the most aggressive single arm.

---

## Section 8 — Risk and Statistical Analysis

9. **`outputs/figures/bootstrap_nav_paths.png`**
   *Figure 9.* Fifty i.i.d.-bootstrapped NAV trajectories (plus the
   median path) of the best-Sharpe strategy (top-10 momentum),
   visualizing the path-ordering uncertainty around the realized
   track record (median terminal NAV ≈ 3.24, 5th percentile ≈ 1.18,
   95th percentile ≈ 9.18).

---

## Optional appendix — Phase 1 smoke test

10. **`outputs/figures/phase1_nav.png`**
    *Figure A1.* NAV of the equal-weight buy-and-hold baseline over
    the full Nasdaq-100 universe (Phase 1 engine smoke test).

11. **`outputs/figures/phase1_drawdown.png`**
    *Figure A2.* Drawdown of the equal-weight buy-and-hold baseline,
    used as a reference passive benchmark for the strategies in
    Sections 4–7.
