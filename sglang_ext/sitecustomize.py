# Loaded automatically by every Python process started with PYTHONPATH=sglang_ext,
# including SGLang's scheduler subprocesses, so the strategies are registered everywhere.
try:
    import forkcache_strategy
    forkcache_strategy.register()
except Exception as e:  # never break interpreter start-up
    import sys
    print(f"[forkcache] eviction strategies not registered: {e}", file=sys.stderr)
