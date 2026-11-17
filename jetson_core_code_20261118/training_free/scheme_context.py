"""Switchable experimental adaptor paths; original vendor modules stay unchanged."""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from factorized_adaptor import FactorizedAdaptor
from compressors import compress_tokens


SCHEMES = {
    'baseline': dict(depth='fixed0', adaptor='native', compression=None, ratio=1.0),
    'cosine50': dict(depth='fixed0', adaptor='native', compression='cosine', ratio=0.5),
    'reorder': dict(depth='fixed0', adaptor='model', compression=None, ratio=1.0),
    'reorder_fp32': dict(depth='fixed0', adaptor='float32', compression=None, ratio=1.0),
    'attention50': dict(depth='fixed0', adaptor='native', compression='attention', ratio=0.5),
    'difference50': dict(depth='fixed0', adaptor='native', compression='difference', ratio=0.5),
    'difference_gate': dict(depth='fixed0', adaptor='native', compression='difference_gate', ratio=0.5),
    'response50': dict(depth='fixed0', adaptor='native', compression='response', ratio=0.5),
    'reorder_attention50': dict(depth='fixed0', adaptor='model', compression='attention', ratio=0.5),
    'reorder_difference50': dict(depth='fixed0', adaptor='model', compression='difference', ratio=0.5),
    'reorder_difference_gate': dict(depth='fixed0', adaptor='model', compression='difference_gate', ratio=0.5),
    'reorder_response50': dict(depth='fixed0', adaptor='model', compression='response', ratio=0.5),
    'skip_fixed1': dict(depth='fixed1', adaptor='native', compression=None, ratio=1.0),
    'skip_adaptive': dict(depth='adaptive', adaptor='native', compression=None, ratio=1.0),
    'skip_local': dict(depth='local', adaptor='native', compression=None, ratio=1.0),
    'reorder_skip_fixed1': dict(depth='fixed1', adaptor='model', compression=None, ratio=1.0),
    'reorder_skip_adaptive': dict(depth='adaptive', adaptor='model', compression=None, ratio=1.0),
    'reorder_skip_local': dict(depth='local', adaptor='model', compression=None, ratio=1.0),
    'attention50_skip_local': dict(depth='local', adaptor='native', compression='attention', ratio=0.5),
    'response50_skip_local': dict(depth='local', adaptor='native', compression='response', ratio=0.5),
    'reorder_attention50_skip_local': dict(depth='local', adaptor='model', compression='attention', ratio=0.5),
    'ar_reference': dict(depth='fixed0', adaptor='native', compression=None, ratio=1.0),
}


def native_with_sizes(adaptor, x, sizes):
    """Original multihead attention with existing proportional-attention correction."""
    b, n, _ = x.shape
    h, d = adaptor.num_heads, adaptor.head_dim
    q = adaptor.q.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
    k = adaptor.k_proj(x).reshape(b, n, h, d).transpose(1, 2)
    v = adaptor.v_proj(x).reshape(b, n, h, d).transpose(1, 2)
    # CUDA fused SDPA backends require bias/query dtype equality. Compute log
    # accurately first, then match q; FP64 debug paths retain FP64 throughout.
    log_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    bias = sizes.to(device=x.device, dtype=log_dtype).log().to(q.dtype).reshape(1, 1, 1, n)
    z = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                     attn_mask=bias, is_causal=False)
    return adaptor.o_proj(z.transpose(1, 2).reshape(b, adaptor.num_q, h*d))


class SchemeRuntime:
    def __init__(self, draft, precisions=('model', 'float32'), difference_threshold=None):
        self.draft = draft
        self.adaptor = draft.imadpt
        self.difference_threshold = difference_threshold
        self.factorized = {p: FactorizedAdaptor(self.adaptor, precision=p) for p in precisions}
        # Scoring directions use frozen weights, with FP32 composition for a stable proxy.
        a = self.adaptor
        q = a.q.detach().float().permute(1, 0, 2)
        w = a.k_proj.weight.detach().float().reshape(a.num_heads, a.head_dim, a.hidden_size)
        self.score_directions = torch.einsum('hqd,hdi->hqi', q, w).reshape(-1, a.hidden_size)
        self.score_directions.div_(math.sqrt(a.head_dim))
        bias = a.k_proj.bias
        self.score_bias = None if bias is None else torch.einsum(
            'hqd,hd->hq', q, bias.detach().float().reshape(a.num_heads, a.head_dim)
        ).reshape(-1) / math.sqrt(a.head_dim)

    def scores(self, x):
        scores = F.linear(x.float(), self.score_directions, self.score_bias)
        return scores

    @contextmanager
    def activate(self, joint, scheme):
        from vispec_controls import DraftVisualContext

        original_prepare = joint.prepare_draft_visual_context
        original_forward = self.adaptor.forward
        state = {'group_sizes': None, 'compression': {}, 'adaptor': scheme['adaptor']}

        def prepare(embeddings, visual_mask, video_grid_thw, spatial_merge_size, m,
                    merge_operator='cosine'):
            method = scheme['compression']
            if method is None or method == 'cosine' or visual_mask is None:
                context = original_prepare(embeddings, visual_mask, video_grid_thw,
                                           spatial_merge_size, m, merge_operator='cosine')
                state['compression'] = dict(method=method or 'identity',
                    original_count=context.original_count, kept_count=context.kept_count,
                    actual_ratio=context.actual_m, group_size_correction=False)
                return context
            locations = visual_mask[0].nonzero(as_tuple=False).flatten()
            x = embeddings[0, locations]
            raw = video_grid_thw[0].tolist() if torch.is_tensor(video_grid_thw) else video_grid_thw
            t, h, w = (int(v) for v in raw)
            grid = (t, h // spatial_merge_size, w // spatial_merge_size)
            scores = self.scores(x) if method in ('attention', 'response') else None
            saliency = scores.softmax(dim=0).amax(dim=1) if method == 'attention' else None
            extra = {'difference_threshold': self.difference_threshold} if method == 'difference_gate' else {}
            tokens, sizes, summary = compress_tokens(x, grid, scheme['ratio'], method=method,
                                                      saliency=saliency, scores=scores, **extra)
            state['group_sizes'] = sizes
            state['compression'] = dict(summary, group_size_correction=True)
            return DraftVisualContext(nominal_m=scheme['ratio'], actual_m=tokens.shape[0]/x.shape[0],
                original_count=x.shape[0], kept_count=tokens.shape[0],
                grid_thw_after_spatial_merge=grid, groups=[], merged_embeddings=tokens,
                notes=['complete_donor_coverage', 'groups_kept_on_device', method], identity=False)

        def forward(x, *args, **kwargs):
            if args or kwargs:
                raise ValueError('Unexpected extra adaptor arguments')
            sizes = state['group_sizes']
            if scheme['adaptor'] == 'native':
                return original_forward(x) if sizes is None else native_with_sizes(self.adaptor, x, sizes)
            return self.factorized[scheme['adaptor']](x, group_sizes=sizes)

        joint.prepare_draft_visual_context = prepare
        self.adaptor.forward = forward
        try:
            yield state
        finally:
            self.adaptor.forward = original_forward
            joint.prepare_draft_visual_context = original_prepare
