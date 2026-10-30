# Akuna Capital Quant Trading Challenge 2026 — Binary Option Market Maker

A Python market-making bot for the Akuna Capital 2026 quant trading challenge. The bot quotes two-sided markets and responds to fill-or-kill (FOK) orders on binary options written on three correlated underlyings: a Fed Funds rate and two equities (Ajarai, Theriodic).

> **Note:** `MarketMaker.py` is the submission file. It runs inside the challenge harness, which supplies `math`, `Underlying`, `BinaryOption`, `MarketParameters`, `Quote`, `FokOrder`, `OrderType`, `Position`, `MarketHistory` and the underlying-ID constants. It does not run on its own.

## Strategy overview

### 1. Parameter estimation (`warm_up`)

- **Rate process:** the Fed Funds rate moves up / down / stays flat with probabilities that tilt toward a target level (mean reversion). `up_probability`, `down_probability` and `reversion_strength` are fitted by **maximum likelihood** using a coarse-to-fine grid search (11³ grid, refined 3 times). Two candidate targets are tested: the default (2.0) and the sample mean. The step size is taken from the median observed move.
- **Equity factor model:** log returns are regressed on rate changes (OLS) to get drift and rate beta. Theriodic's residual is treated as the sector factor; Ajarai's sector beta is `Cov(resid_A, resid_T) / Var(resid_T)`, and its idiosyncratic volatility is the remaining variance.
- **Shrinkage:** each estimate is blended with a generic prior, weighted by `min(n_days / 20, 1)`, so short histories don't produce extreme parameters. Histories under 10 days set a low-confidence flag that widens spreads and cuts size.

### 2. Pricing

- Theoretical value = **Monte Carlo** probability that the binary pays out, simulated forward with the estimated parameters.
- The number of paths scales with time to expiry and the number of live options, to stay within the time budget. Prices are cached per (option, expiry, market state).

### 3. Quoting (`quote`)

- **Avellaneda–Stoikov reservation price:** `theo − γ · q_eff · θ(1−θ) · τ`, where `θ(1−θ)` is the Bernoulli payoff variance and `τ` is steps to expiry.
- **Correlated inventory:** `q_eff` adds 30% of the exposure from other options on the same underlyings.
- **Spread** = base half-spread × confidence × counterparty-toxicity × `sqrt(τ / 4)` multipliers. The side that would grow the position is widened further.
- **Sizing:** a fraction of cash per quote, capped by `BASE_QTY / sqrt(#options)`. A capped boost applies when starting capital is small. Below a critical cash level the bot quotes 0.01 / 0.99.

### 4. FOK orders (`respond_to_fok`)

An order is accepted only when its edge against theo exceeds a margin (scaled by confidence, toxicity and expiry) and the capital it needs fits within a per-order budget.

### 5. Counterparty toxicity

After each fill, the next theo is compared with the trade price (markout). Adverse markouts update an exponentially weighted toxicity score per counterparty, which widens spreads, raises the FOK margin and reduces size for that counterparty.

### 6. Robustness

Every entry point is wrapped in `try/except` with a safe fallback (theo 0.5, very wide quotes, rejecting FOKs, or generic parameters), so one error never crashes the bot.

## Key parameters

| Parameter | Value | Role |
|---|---|---|
| `BASE_HALF_SPREAD` | 0.02 | Base half-spread |
| `INVENTORY_GAMMA` | 0.01 | Inventory risk aversion (γ) |
| `POSITION_SOFT_CAP` / `WIDEN_K` | 25 / 1.0 | Spread widening with inventory |
| `BASE_QTY` | 60 | Base quote size |
| `FOK_EDGE_MARGIN` | 0.02 | Minimum edge to accept a FOK |
| `CAPITAL_FRACTION_PER_QUOTE` / `_FOK` | 0.15 / 0.20 | Capital budget per trade |
| `TOXICITY_SPREAD_SCALE` | 5.0 | Toxicity → spread multiplier |
| `CORRELATION_BLEND` | 0.3 | Weight on correlated exposure |
| `SHRINKAGE_HORIZON_DAYS` | 20 | Days of history before full trust in estimates |
| `CAPITAL_BOOST_REFERENCE` / `MAX_CAPITAL_BOOST` | 40 / 5.0 | Small-capital size boost |

## Tech

Pure Python (standard library only): MLE grid search, OLS, Monte Carlo simulation and Avellaneda–Stoikov market making.

## Author

Junhao (Shelby) Lin — BASc Mathematics, University College London · [LinkedIn](https://www.linkedin.com/in/junhao-shelby-lin-a443903a6/)
