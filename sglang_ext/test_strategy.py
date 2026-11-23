"""Logic test for the SGLang strategies with fake tree nodes (no SGLang or GPU needed).

  uv run python sglang_ext/test_strategy.py
On CARC, with SGLang installed, it imports the real EvictionStrategy base class instead.
"""
import os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sglang.srt.mem_cache.evict_policy  # noqa: F401  real base class (CARC)
    REAL = True
except Exception:
    REAL = False
    from abc import ABC, abstractmethod
    class EvictionStrategy(ABC):
        @abstractmethod
        def get_priority(self, node): ...
    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
        sys.modules[name] = types.ModuleType(name)
    m = types.ModuleType("sglang.srt.mem_cache.evict_policy")
    m.EvictionStrategy = EvictionStrategy
    sys.modules[m.__name__] = m

from forkcache_strategy import DistAwareStrategy, SkiRentalStrategy, _prefix_len

def node(parent, n, age):
    return types.SimpleNamespace(key=[0] * n, parent=parent, last_access_time=time.monotonic() - age)

root = types.SimpleNamespace(key=None, parent=None, last_access_time=0)
a = node(root, 1000, 0); b = node(a, 500, 5.0); c = node(a, 500, 50.0)
assert _prefix_len(b) == 1500
s = SkiRentalStrategy(b=1e-4, c=0.0, pool_tokens=100_000)        # tau = b * pool = 10 s
pb, pc = s.get_priority(b), s.get_priority(c)
assert pb[0] == 1 and pc[0] == 0, (pb, pc)                        # idle 5 s pinned, 50 s expired
assert min(pb, pc) == pc                                          # expired evicted first
d = DistAwareStrategy([[1200, 0.0], [1e12, 30.0]])
assert d.get_priority(a)[0] == 0 and d.get_priority(b)[0] == 1 and d.get_priority(c)[0] == 0
print(f"strategy logic ok (base class: {'real SGLang' if REAL else 'stand-in'})")
