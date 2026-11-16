# Notice of changes to the Forecasting Engine, November 2026

*Draft for Toby to send to Dr Catalin Burlacu, Alpha Norm. Under the Change
Management Plan, each change below is notified with an objection window.*

Dear Dr Catalin,

Since UAT 3 (1 Nov 2026) we have made seven changes to how the engine scores
models and builds the portfolio, and rerun every result on a longer history. Each
change below says what changed, why, and how the results moved. Each was decided
on design grounds before its results were seen, except where we say otherwise.
The full figures are in the repository's `docs/evidence/2026-11-16-full-history.md`.

**1. What the Signal Rank IC measures against.** The Signal Rank IC scores what
a model adds beyond its average forecast. Until now it subtracted the training
window's average return from every model's forecast. It now subtracts each
model's own average forecast. The old version counted any gap between those two
averages as skill: a model that forecast zero every day scored +0.21 with t 3.0
on the Sep 2026 data and passed both gates. On that data the research equity
model moved from 0.259 to 0.237, and the derived equity polynomial from 0.166 to
0.156. A model that forecasts the same value throughout now scores exactly zero.

**2. The standard error behind the t-statistic.** At a 20-day horizon,
neighbouring forecasts share 19 of their 20 days, so their errors overlap. The old
standard error (Newey-West) understated that overlap. In a simulation with no
predictability, it let t exceed 2 about 4.6% of the time instead of 2.3%, so
"t above 2" acted more like "t above 1.7". The gate now uses a standard error that
counts the overlap in full (Hansen-Hodrick), and never one smaller than the old
one. In the same simulation t exceeds 2 in 2.9% of cases. The t-statistic in the
trust gate inside the optimiser uses the same standard error. Some t-statistics
near 2 are now lower.

**3. A research model for the bond index.** There was a research model for
equity but none for bond. We added one, labelled as research, on seven fixed
inputs: rates, spreads and volatility gauges, read as levels. It is corrected for
a known small-sample bias (Stambaugh bias; Amihud & Hurvich 2004). Its inputs were
chosen on the Aug 2016 – Sep 2026 data, so its score there is optimistic. On data
it was not chosen on, before Aug 2016, it scores −0.004 (t −0.05) at the default
20-day horizon. It does not pass the gate, and we have not used it in the
portfolio.

**4. History from 2006.** We extended the Bloomberg data back to 31 Aug 2006,
the earliest our access allowed. That more than doubles the scored period, from 8 to 18
years, and includes the 2008 crisis as a genuine test: the first test window
opens in October 2008. No window rule changed. It also gave both research models
a test on years their inputs were not chosen on (before Aug 2016), and both score
about zero there. On the longer history:
- the derived equity polynomial passes the gate at the 1-, 5- and 10-day horizons;
- at the default 20-day horizon it narrowly fails (Signal Rank IC +0.103, t 1.98,
  against +0.156 and t 2.31 on the Sep 2026 data);
- no bond model passes at any horizon;
- at 20 days, no model passes for either index.

**5. The long-run equity premium.** The optimiser infers each risk level's risk
aversion from a long-run equity-minus-bond premium. That premium was the plain
average of every day to date. With data starting in 2006, that average would rest
on two years ending in the 2008 crash, and would very likely have turned negative
in 2008–09. That would have switched the forecasts off, or made the tilts
explode. The premium now blends the data's own average with a published estimate:
6.6% a year, US large-company stocks over long-term government bonds, 1926–2004
(Goetzmann & Ibbotson 2005). The published estimate counts as 30 years of data,
the weight at which it is uncertain by about ±3 points a year. Over the backtest
the premium now stays between 4.2% and 7.9% a year and is never zero or below.
The limitations:
- the published estimate's bond index is long-term government bonds, so it
  probably understates our premium by about 0.4 points a year;
- by 2026 the published estimate still carries about 60% of the weight.

*Question for you:* would Alpha Norm prefer its own long-run premium assumption
in place of the published 6.6%?

**6. The no-trade band is removed.** This changes CTQ v5, Module 3, which
requires the allocation to be "held through a cost-aware no-trade band". The band
let a rebalance trade only part of the way to its target, to save costs. Its
width was expressed in horizon terms, so it grew as the horizon shortened:
- at the 1-day horizon it was about four times the whole tilt range, which froze
  the portfolio (turnover 0–3% a year);
- at the 5-day horizon it let a riskier level hold less equity than a safer one
  at 22 rebalances.

Every rebalance now trades straight to its target and pays your 3 bp (equity)
and 5 bp (bond) on what it trades. Costs are at most 0.06% a year at any level
and horizon, so the band saved at most that, and no riskier level now ever holds
less equity than a safer one. Toby confirmed the removal after seeing these
results; nothing was changed because of them.

**7. Fixes from the optimiser audit.**
- **Directional P&L:** the long/cash comparison and its hit rate now follow each
  model's signal rather than its raw forecast. A model whose average forecast is
  positive used to stay invested almost every day, whatever its signal said.
- **Charts:** each multi-day return is now plotted when it was earned, at the end
  of its holding period.
- **Labels:** the Portfolio page now distinguishes the signals the weights were
  set from and the full forecasts the Monte Carlo simulation is centred on.
- **Documentation:** we now explain why strong signals put every risk level on
  the same bound, why level 5 sits at the 20% floor, and why the latest weights
  can trail the data by up to about 7 weeks.

**What the results say now.** On the longer history, net of your costs, the risk
levels behave as designed: net Sharpe rises from about 0.79 at level 1 to about
1.04 at level 5, against 0.88 for 50/50. But the forecasts add no value at any
horizon. Against the same risk level without forecasts, t is between −1.38 and
+0.05, and never significant. We report this as a known limitation, not a
delivery failure, as the CTQ provides.

Two more disclosures:
- **The forecast-trust gate is re-checked at every rebalance.** A model with no
  skill can therefore pass it at some point far more often than the 2.3% one
  check allows. A simulation on our data estimates 11% of the time at 1 day, 18%
  at 5 days and 35% at 20 days.
- **Why the gate opened less than at UAT 3.** At UAT 3 the gate opened at 26 of
  99 rebalances; on the longer history, with the derived polynomial for both
  indices, it opens at 3 of 226. That difference comes mostly from the choice of
  bond model, not from the longer history. In no combination did the forecasts
  add value.

Kind regards,
Toby Chan, for Team Finlytics
