"""
SGLang eviction strategies for idle-agent KV (Phase 4.1).

Plugs into SGLang's pluggable radix eviction (mem_cache/evict_policy.py) without
forking it. A node whose KV has been idle for less than its keep threshold is
"pinned": every unpinned node is evicted first (LRU), and pinned nodes go only
when nothing else is left, earliest deadline first. This matches budget mode of
forkcache-sim's ski_rental policy, so simulator and server can be compared.

  --radix-eviction-policy skirental
  --radix-eviction-policy-config '{"b": 5.1e-5, "c": 1.2e-10, "pool_tokens": 300000}'

  --radix-eviction-policy distaware
  --radix-eviction-policy-config '{"thresholds": [[1024, 12.0], [4096, 30.0], [1e12, 0.0]]}'

b, c: prefill curve R(L) = a + bL + cL^2 in seconds (bench/prefill_curve.py; a is unused).
pool_tokens: KV pool size (SGLang logs it as max_total_num_tokens); sets c_mem = 1/pool_tokens.
thresholds: [[L_upper, theta_seconds], ...] fitted offline (theta 0 = never pin).
"""

from __future__ import annotations

import bisect
import time
from typing import Tuple

from sglang.srt.mem_cache.evict_policy import EvictionStrategy


def _prefix_len(node) -> int:
    """Tokens from the root to the end of this node (the context an idle agent holds)."""
    n, total = node, 0
    while n is not None and n.key is not None:
        total += len(n.key)
        n = n.parent
    return total


class SkiRentalStrategy(EvictionStrategy):
    """Pin for tau(L) = R(L) / (c_mem L) after the last access, then plain LRU."""

    def __init__(self, b: float, c: float = 0.0, pool_tokens: int = 1, a: float = 0.0):
        self.b, self.c, self.pool = float(b), float(c), float(pool_tokens)

    def tau(self, L: int) -> float:
        if L <= 0:
            return 0.0
        return (self.b * L + self.c * L * L) * self.pool / L   # = (b + cL) * pool_tokens

    def get_priority(self, node) -> Tuple[int, float]:
        deadline = node.last_access_time + self.tau(_prefix_len(node))
        if time.monotonic() < deadline:
            return (1, deadline)          # pinned: evicted last, earliest deadline first
        return (0, node.last_access_time)  # expired: LRU


class DistAwareStrategy(EvictionStrategy):
    """Pin for a fitted threshold theta(L) looked up from an offline table."""

    def __init__(self, thresholds):
        rows = sorted((float(u), float(t)) for u, t in thresholds)
        self.uppers = [u for u, _ in rows]
        self.thetas = [t for _, t in rows]

    def theta(self, L: int) -> float:
        i = bisect.bisect_left(self.uppers, L)
        return self.thetas[min(i, len(self.thetas) - 1)]

    def get_priority(self, node) -> Tuple[int, float]:
        th = self.theta(_prefix_len(node))
        deadline = node.last_access_time + th
        if th > 0 and time.monotonic() < deadline:
            # Furthest predicted return evicted first among pinned (per-agent Belady).
            return (1, -deadline)
        return (0, node.last_access_time)


def register() -> None:
    from sglang.srt.mem_cache import utils
    utils._EVICTION_POLICY_FACTORIES.setdefault("skirental", SkiRentalStrategy)
    utils._EVICTION_POLICY_FACTORIES.setdefault("distaware", DistAwareStrategy)
    try:
        from sglang.srt.arg_groups import choices
        for name in ("skirental", "distaware"):
            if name not in choices.RADIX_EVICTION_POLICY_CHOICES:
                choices.RADIX_EVICTION_POLICY_CHOICES.append(name)
    except ImportError:
        pass
