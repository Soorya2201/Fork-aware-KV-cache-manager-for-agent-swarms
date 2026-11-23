"""
kernels/shared_prefix_attn.py — decode attention for forks that share a prefix (Phase 5.1).

G fork groups; each group has F forks that share a prefix of P tokens and each
fork has its own suffix (length S_f <= S_max). One decode query per fork.

  out_f = softmax(q_f [K_prefix; K_suffix_f]^T / sqrt(d)) [V_prefix; V_suffix_f]

Shared-prefix trick (same idea as FlashInfer cascade attention): attention over
the prefix is computed for all F forks in one pass, so the prefix K/V is read from
HBM once per group instead of once per fork; each fork's suffix is attended
separately; the two partial softmaxes are merged exactly with log-sum-exp.

  merge: m = max(m1, m2);  l = l1 e^(m1-m) + l2 e^(m2-m)
         o = (o1 l1 e^(m1-m) + o2 l2 e^(m2-m)) / l

Layout (GQA, Hq = Hk * R):
  q        [G, F, Hq, D]
  k_pre    [G, P, Hk, D]     v_pre same
  k_suf    [G, F, S_max, Hk, D]   v_suf same, suf_len [G, F] int32

`attention_reference` and `attention_decomposed_torch` run anywhere (tests on a Mac
CPU); `shared_prefix_attention` needs CUDA + Triton (CARC).
"""

from __future__ import annotations

import math

import torch


# ----------------------------------------------------------------------------- references
def attention_reference(q, k_pre, v_pre, k_suf, v_suf, suf_len):
    """Naive attention, prefix duplicated per fork. Returns [G, F, Hq, D] (fp32)."""
    G, F, Hq, D = q.shape
    Hk = k_pre.shape[2]
    R = Hq // Hk
    out = torch.empty(G, F, Hq, D, dtype=torch.float32, device=q.device)
    for g in range(G):
        for f in range(F):
            s = int(suf_len[g, f])
            K = torch.cat([k_pre[g], k_suf[g, f, :s]], 0).float()   # [L, Hk, D]
            V = torch.cat([v_pre[g], v_suf[g, f, :s]], 0).float()
            K = K.repeat_interleave(R, dim=1)                         # [L, Hq, D]
            V = V.repeat_interleave(R, dim=1)
            sc = torch.einsum("hd,lhd->hl", q[g, f].float(), K) / math.sqrt(D)
            out[g, f] = torch.einsum("hl,lhd->hd", sc.softmax(-1), V)
    return out


def _partial(qf, K, V, scale):
    """Partial softmax pieces for queries qf [N, D] over K, V [L, D]: (m, l, o)."""
    sc = (qf @ K.T) * scale
    m = sc.max(-1).values
    p = torch.exp(sc - m[:, None])
    l = p.sum(-1)
    o = (p @ V) / l[:, None]
    return m, l, o


def merge(m1, l1, o1, m2, l2, o2):
    m = torch.maximum(m1, m2)
    a1, a2 = l1 * torch.exp(m1 - m), l2 * torch.exp(m2 - m)
    l = a1 + a2
    return m, l, (o1 * a1[..., None] + o2 * a2[..., None]) / l[..., None]


def attention_decomposed_torch(q, k_pre, v_pre, k_suf, v_suf, suf_len):
    """Prefix once per group (all forks' queries together), suffix per fork, LSE merge."""
    G, F, Hq, D = q.shape
    Hk = k_pre.shape[2]
    R = Hq // Hk
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(G, F, Hq, D, dtype=torch.float32, device=q.device)
    for g in range(G):
        for h in range(Hk):
            # All forks x the R query heads of this KV head: one pass over the prefix.
            qg = q[g, :, h * R:(h + 1) * R].reshape(F * R, D).float()
            mp, lp, op = _partial(qg, k_pre[g, :, h].float(), v_pre[g, :, h].float(), scale)
            mp, lp, op = mp.view(F, R), lp.view(F, R), op.view(F, R, D)
            for f in range(F):
                s = int(suf_len[g, f])
                if s == 0:
                    out[g, f, h * R:(h + 1) * R] = op[f]
                    continue
                ms, ls, os_ = _partial(q[g, f, h * R:(h + 1) * R].float(),
                                       k_suf[g, f, :s, h].float(), v_suf[g, f, :s, h].float(), scale)
                _, _, o = merge(mp[f], lp[f], op[f], ms, ls, os_)
                out[g, f, h * R:(h + 1) * R] = o
    return out


# ----------------------------------------------------------------------------- Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # Mac: no Triton wheel
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _prefix_kernel(Q, K, V, M_out, L_out, O_out,
                       P, FR, D: tl.constexpr, R: tl.constexpr, HK, CHUNK,
                       sqg, sqf, sqh, skg, skp, skh, smg, smh, smc,
                       scale,
                       BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr):
        """One program = (group g, kv head h, prefix chunk c); queries = all F*R rows."""
        g = tl.program_id(0)
        h = tl.program_id(1)
        c = tl.program_id(2)
        rq = tl.arange(0, BLOCK_Q)                 # row = f * R + r
        rd = tl.arange(0, D)
        f_idx = rq // R
        r_idx = rq % R
        qmask = rq < FR
        q_ptr = Q + g * sqg + f_idx[:, None] * sqf + (h * R + r_idx)[:, None] * sqh + rd[None, :]
        q = tl.load(q_ptr, mask=qmask[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, D], tl.float32)
        start = c * CHUNK
        end = tl.minimum(start + CHUNK, P)
        for n0 in range(start, end, BLOCK_N):
            rn = n0 + tl.arange(0, BLOCK_N)
            nmask = rn < end
            kv_off = g * skg + rn[:, None] * skp + h * skh + rd[None, :]
            k = tl.load(K + kv_off, mask=nmask[:, None], other=0.0).to(tl.float32)
            v = tl.load(V + kv_off, mask=nmask[:, None], other=0.0).to(tl.float32)
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(nmask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_new
        o = acc / l_i[:, None]
        base = g * smg + h * smh + c * smc
        tl.store(M_out + base + rq, m_i, mask=qmask)
        tl.store(L_out + base + rq, l_i, mask=qmask)
        tl.store(O_out + (base + rq)[:, None] * D + rd[None, :], o, mask=qmask[:, None])


def shared_prefix_attention(q, k_pre, v_pre, k_suf, v_suf, suf_len, chunk: int = 1024,
                            block_n: int = 64):
    """Triton prefix pass (prefix K/V read once per group) + torch suffix pass + LSE merge.

    The suffix pass is per fork and short; it uses torch SDPA-style math here and is
    the next thing to move into Triton.
    """
    assert HAS_TRITON and q.is_cuda, "needs CUDA + Triton"
    G, F, Hq, D = q.shape
    P, Hk = k_pre.shape[1], k_pre.shape[2]
    R = Hq // Hk
    FR = F * R
    n_chunks = triton.cdiv(P, chunk)
    BLOCK_Q = max(16, triton.next_power_of_2(FR))
    M = torch.empty(G, Hk, n_chunks, FR, device=q.device, dtype=torch.float32)
    L = torch.empty_like(M)
    O = torch.empty(G, Hk, n_chunks, FR, D, device=q.device, dtype=torch.float32)
    q_, k_, v_ = q.contiguous(), k_pre.contiguous(), v_pre.contiguous()
    _prefix_kernel[(G, Hk, n_chunks)](
        q_, k_, v_, M, L, O, P, FR, D, R, Hk, chunk,
        q_.stride(0), q_.stride(1), q_.stride(2), k_.stride(0), k_.stride(1), k_.stride(2),
        M.stride(0), M.stride(1), M.stride(2), 1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q, BLOCK_N=block_n)
    # Merge prefix chunks.
    m = M.max(dim=2).values
    w = L * torch.exp(M - m[:, :, None])
    lp = w.sum(2)
    op = (O * w[..., None]).sum(2) / lp[..., None]          # [G, Hk, FR, D]
    op = op.view(G, Hk, F, R, D).permute(0, 2, 1, 3, 4).reshape(G, F, Hq, D)
    mp = m.view(G, Hk, F, R).permute(0, 2, 1, 3).reshape(G, F, Hq)
    lp = lp.view(G, Hk, F, R).permute(0, 2, 1, 3).reshape(G, F, Hq)
    # Suffix per fork (masked), then merge.
    Smax = k_suf.shape[2]
    ks = k_suf.float().repeat_interleave(R, dim=3)              # [G, F, S, Hq, D]
    vs = v_suf.float().repeat_interleave(R, dim=3)
    sc = torch.einsum("gfhd,gfshd->gfhs", q.float(), ks) / math.sqrt(D)
    mask = torch.arange(Smax, device=q.device)[None, None, :] < suf_len[..., None]
    sc = sc.masked_fill(~mask[:, :, None, :], float("-inf"))
    ms = sc.max(-1).values
    has = mask.any(-1)[..., None]                               # forks with empty suffix
    ms = torch.where(has, ms, torch.full_like(ms, float("-inf")))
    p = torch.exp(sc - torch.where(has, ms, torch.zeros_like(ms))[..., None])
    ls = p.sum(-1)
    os_ = torch.einsum("gfhs,gfshd->gfhd", p, vs) / ls.clamp(min=1e-30)[..., None]
    ms = torch.where(has, ms, torch.full_like(ms, -1e30))
    ls = torch.where(has, ls, torch.zeros_like(ls))
    _, _, o = merge(mp, lp, op, ms, ls, os_)
    return o
