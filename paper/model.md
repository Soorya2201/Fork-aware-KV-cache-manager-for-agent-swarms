# Keep, compress or evict: a cost model for idle agent KV

## 1. Setup

An agent finishes an LLM call holding **L** tokens of KV cache, then goes idle (waiting on a tool
or on other agents) for an unknown time **T** before its next call reuses that prefix.

- **Keep:** memory rent of `r = c_mem · L` per second. We set `c_mem = 1 / N_pool` GPU-seconds
  per token-second, where `N_pool` is the KV pool size in tokens: holding the whole pool for
  one second costs one GPU-second, because memory held by an idle agent is memory a running
  request cannot batch into.
- **Evict:** pay a one-time recompute `B = R(L)` GPU-seconds when the agent returns, where
  `R(L) = a + bL + cL²` is the measured prefill curve (`bench/prefill_curve.py`). The per-call
  overhead `a` is paid on return either way, so we use `B = bL + cL²`.

A policy chooses a threshold **θ**: keep for up to θ seconds, then evict. Its cost on gap T is

```
C(θ, T) = r · min(T, θ) + B · 1[T > θ]
```

The offline optimum knows T and pays `OPT(T) = min(rT, B)`. The break-even time is

```
τ(L) = B / r = R(L) / (c_mem · L)
```

This is ski rental: renting (keeping) costs r per unit time, buying (evicting, then
recomputing) costs B once.

## 2. Deterministic threshold θ = τ is 2-competitive, and no deterministic policy does better

**Upper bound.** If T ≤ τ, the policy keeps throughout: `C = rT = OPT`, since `rT ≤ rτ = B`.
If T > τ, it pays `rτ + B = 2B`, while `OPT = B`. So `C ≤ 2·OPT` for every T.

**General θ.** For an arbitrary threshold the worst gap is just above θ:

```
ratio(θ) = sup_T C(θ,T)/OPT(T) = 1 + max(τ/θ, θ/τ)
```

(For θ < τ, T ↓ θ gives `(rθ + B)/(rθ) = 1 + τ/θ`. For θ > τ, T ↓ θ gives
`(rθ + B)/B = 1 + θ/τ`.) This is minimized at θ = τ with value 2, so no deterministic
threshold beats 2. ∎

**Consequence used later.** Restricting θ to `[τ/(α−1), (α−1)τ]` guarantees a worst-case
ratio of at most α for any α ≥ 2. That is the robustness knob of the distribution-aware
policy (`dist_aware_agent_robust3` uses α = 3, so θ ∈ [τ/2, 2τ]).

## 3. Randomized threshold: e/(e−1) ≈ 1.582

Draw `θ = τX` with X having density `p(x) = eˣ/(e−1)` on [0, 1] (inverse CDF
`x = ln(1 + u(e−1))`). Write `s = T/τ`.

*Case s ≤ 1.*
```
E[C]/B = ∫₀ˢ (x+1) p(x) dx + s ∫ₛ¹ p(x) dx
       = [x eˣ]₀ˢ/(e−1) + s(e − eˢ)/(e−1)
       = s·e/(e−1)
```
using `∫(x+1)eˣ dx = x eˣ`. Since `OPT = rT = sB`, the ratio is `e/(e−1)`.

*Case s > 1.* Every threshold fires before T:
`E[C]/B = ∫₀¹ (x+1)p(x) dx = [x eˣ]₀¹/(e−1) = e/(e−1)`, and `OPT = B`.

The ratio is exactly `e/(e−1)` for every T, which is optimal for randomized ski rental
(Karlin et al., 1994). ∎ Both bounds are checked numerically in
`sim/forkcache-sim/tests/known_answers.rs`.

## 4. Our contribution: a distribution-aware threshold fitted to the measured gaps

Worst-case bounds ignore what we measured: the idle-gap distribution F of the real workload.
For a known F, the expected cost of threshold θ is

```
J(θ) = r ∫₀^θ (1 − F(t)) dt + B (1 − F(θ))
J'(θ) = (1 − F(θ)) · (r − B·h(θ)),    h = f/(1 − F) the hazard rate.
```

So stationary points are where the **hazard rate crosses 1/τ**:

- **Decreasing hazard** (heavy tails: "the longer it has been idle, the less likely it
  returns soon", typical of tool calls with stragglers). J falls and then rises: there is an
  interior optimum `h(θ*) = 1/τ`. Keeping for a while and then giving up is right.
- **Increasing hazard** (gaps concentrated around a typical value, as in turn-based agents
  where a gap is "one round of the other agents"). The stationary point is a *maximum*, so the
  optimum is at an endpoint: evict immediately (`J(0) = B`) or keep until the agent returns
  (`J(∞) = r·E[T]`). The rule becomes **keep iff E[T] < τ(L)**.

**Empirical version (what the simulator runs).** With observed gaps `t₁ ≤ … ≤ tₙ`, the
empirical J(θ) rises linearly between samples and jumps down at each sample, so

```
θ* ∈ {0, t₁, …, tₙ}
```

and the exact minimizer takes one pass over the sorted gaps with prefix sums: O(n log n) to
fit, O(n) per decision (`policies/dist_aware.rs::best_threshold`, checked against brute force
in the tests).

**Conditioning.** The gap depends on *who* is idle: in a turn-based system an agent's turn
position fixes most of its wait. `dist_aware_agent` fits F separately per agent. To keep the
evaluation honest, F for a call in session s is fitted only on the *other* sessions
(leave-one-session-out). The policy never sees the gap it is deciding about.

**Robustness.** A fitted F can be wrong (distribution shift, a new tool). Clamping θ* to
`[τ/(α−1), (α−1)τ]` keeps the worst-case ratio ≤ α while staying near the fitted optimum
when F is right. This is the consistency–robustness trade-off of learning-augmented ski rental
(Purohit, Svitkina & Kumar, NeurIPS 2018).

## 5. Memory pressure: many idle agents at once

With a pool of `N_pool` tokens, the set of kept agents must fit:

```
minimize   Σᵢ E[Cᵢ(θᵢ)]
subject to Σᵢ Lᵢ · 1[agent i kept at time t] ≤ N_pool − (active KV)   for all t
```

Relaxing the constraint with a multiplier λ ≥ 0 (the shadow price of memory) gives
independent per-agent problems with rent `(c_mem + λ)·Lᵢ`. **Pressure behaves like a higher
memory price**: every τᵢ shrinks by the factor `c_mem/(c_mem + λ)`, and λ is set so that
expected occupancy matches the pool. Online, when the pool is full, the engine must unpin
someone. Among pinned agents, the one whose predicted return is furthest away frees memory
for the longest at the same recompute price, which is a per-agent Belady rule. That is what
the predictive policies (`dist_aware*`, `continuum_style`) do in `on_pressure`. The
non-predictive ski-rental and TTL policies evict the earliest deadline.

A turn-based workload makes one failure obvious: **LRU evicts exactly the agent that speaks
next** (the one idle longest), so on a round-robin pattern larger than the pool it gets zero
hits. The hand-made test `lru_thrashes_on_round_robin_and_belady_does_not` checks this.

### 5.3 Fork-aware: keep only the prefix the next call will reuse

Sections 2–4 treat an idle agent's KV as one item of size L. It is not. The agent's next
prompt forks off the old context at some point K ≤ L, and only the blocks before K can hit.
In the measured workload the app rebuilds a windowed history every turn, so the next call
reuses a median of 48% of the old context (P10 35%, P90 77%), and the fraction is stable per
round (standard deviation about 0.1). Keeping the tail past K pays rent and saves nothing.

The decision therefore has two parts.

1. **Where to cut (K).** With R(K→K') = b(K'−K) + c(K'²−K²) the cost of recomputing tokens
   K..K' on top of a cached prefix of K tokens, and T the gap, keeping a prefix of k tokens
   costs, in expectation over the fork point K*,

   ```
   F(k) = c_mem · k · E[T]  +  E[ R(k → K*) · 1[K* > k] ].
   ```

   The first term grows linearly and the second is a decreasing convex function of k, so this
   is a newsvendor problem. The minimum lies at k = 0, at k = L, or at one of the observed fork
   points. It is fitted per (agent, round) on the other sessions only (leave one session out).

2. **How long to keep it.** Apply any threshold rule from sections 2–4 to the kept prefix,
   with its marginal costs. Of the kept prefix, the part P′ that siblings also hold stays
   cached anyway, so

   ```
   rent = c_mem (k − P′),   buy = R(P′ → k).
   ```

   The guarantees of sections 2–3 carry over to the kept prefix unchanged, because they are
   statements about any rent/buy pair.

The tail is released at once. The policy `fa_<rule>` in the simulator does this.
`fa_oracle_true_fork` knows the true K* and the true gap, and lower-bounds it.

On the Mac trace, this step accounts for the whole gain over the earlier
`dist_aware_agent`. With full-context costs, even the per-episode oracle loses to a 10 s TTL
at large pools. The TTL wins because it drops the dead tail early, not because it predicts
anything. In this workload agents of the same session share almost no radix blocks (each
prompt starts with its own system prompt), so P′ ≈ 0. P′ matters for engines and apps where
siblings share a common transcript prefix.

### 5.4 Budget mode: a fixed pool has no memory price

When the pool is a fixed budget and unused memory is simply idle, pricing memory at c_mem
answers the wrong question. The constraint is what matters, and λ in section 5 is whatever
keeps the pool full. The policies therefore switch rules in budget mode:
- Pin every idle agent until a safe bound on its predicted return (1.2 × the largest gap seen
  for that agent in other sessions).
- Keep every block the next call might reuse (the largest fork point seen), and drop only the
  tail that is certainly dead.
- When the pool is full, unpin the agent predicted to return last.

In a turn-based app the return order is almost deterministic, so this approximates Belady.
On the CARC V100 trace at a 1 GB pool, hit rate rises from 6% (LRU) to 30%, against 37% for
Belady.

On that GPU, prefill runs at about 5,400 tokens/s, so at the full 12.7 GB pool the break-even
time τ is about 44 s. That is shorter than the 56 s it takes the other 12 agents to speak. When
memory has a price, evicting immediately is optimal for this app, and the distribution-aware
threshold learns it: its cost equals the oracle's. Deterministic ski rental waits τ and then
evicts anyway, which costs 1.8×, close to its worst case of 2.

## 6. Third option: compress (Phase 5)

Quantizing the idle KV to INT8 (or FP8) halves the rent (`r/2` relative to fp16) at a one-time
cost `Q(L)` (quantize and dequantize, memory-bound and cheap next to prefill) plus a risk of
changed outputs. With three options the per-agent decision becomes a two-threshold rule: keep
fp16 until θ₁, then compress until θ₂, then evict. The same hazard-rate analysis applies to
each switch. Accuracy is checked separately (`compression/kv_int8_check.py`) because a rent
saving that silently changes the reasoning trajectory is not a saving.

## 7. What the model leaves out (stated so we can test it)

- Recompute is charged at the measured single-request prefill curve. Batching can make
  recompute cheaper (or, under load, more disruptive) than R(L).
- c_mem is an accounting price, not a measured throughput loss. Phase 3 measures the real
  trade-off (TTFT and throughput on SGLang) and checks the simulator within about 10–15%.
- The simulator keeps each session's recorded timing. It models the cache, not queueing.
