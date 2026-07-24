"""Relative-position encodings for neighbour pairs (the ablation switch).

The interaction network V sees, per neighbour slot, [x_i, x_k, <rel features>].
This module builds the <rel features> part. Modes:

  "none"        nothing appended (V is blind to where the neighbour sits)
  "offset"      (dr/pad, dc/pad)                       <- repo default, unchanged
  "offset_dist" (dr/pad, dc/pad, dist / (pad*sqrt(2))) distance normalised to (0, 1]
  "embed"       a learned emb_dim vector per discrete offset slot, indexed by
                (dr+pad)*K + (dc+pad); transformer-style relative embedding.
                Strictly more expressive than any hand-crafted scalar, still
                size-invariant (depends only on the offset, never on N).

Angle note: angle+distance is just the polar form of (dr, dc), so it adds no
information; and a raw angle has a wrap discontinuity (179 vs -181 degrees).
(cos t, sin t) = (dc, dr)/dist is the safe angular encoding and is already
spanned by "offset"+"offset_dist", so no separate angle mode is provided.
"""
import math

import torch
import torch.nn as nn

MODES = ("none", "offset", "offset_dist", "embed")


class RelEncoding(nn.Module):
    def __init__(self, mode="offset", K=3, emb_dim=8):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"rel_encoding must be one of {MODES}, got {mode!r}")
        self.mode, self.K = mode, K
        self.pad = K // 2
        self.denom = float(max(self.pad, 1))
        self.diag = self.denom * math.sqrt(2.0)
        self.table = nn.Embedding(K * K, emb_dim) if mode == "embed" else None

    @property
    def extra_dim(self):
        return {"none": 0, "offset": 2, "offset_dist": 3,
                "embed": self.table.embedding_dim if self.table else 0}[self.mode]

    def forward(self, di, dj, B, device, dtype):
        """Features for one neighbour slot at offset (di, dj) -> (B, extra_dim) or None."""
        if self.mode == "none":
            return None
        if self.mode == "embed":
            idx = torch.tensor((di + self.pad) * self.K + (dj + self.pad), device=device)
            return self.table(idx).to(dtype).unsqueeze(0).expand(B, -1)
        vals = [di / self.denom, dj / self.denom]
        if self.mode == "offset_dist":
            vals.append(math.sqrt(di * di + dj * dj) / self.diag)
        return torch.tensor(vals, device=device, dtype=dtype).expand(B, len(vals))
