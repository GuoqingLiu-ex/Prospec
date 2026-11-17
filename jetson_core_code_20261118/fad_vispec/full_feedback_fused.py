"""One-CTA same-semantics feedback candidate; CUDA execution is not yet proven.

Only original host mask/retrieve and already-known submitted/stop IDs are read
on CPU. Runtime scalar kernel arguments replace many Tensor comparison launches.
They have a real parameter payload cost; no zero-transfer/performance claim is
made. No Target/argmax, host device-value read, or Torch fallback is added.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

SCHEMA = 'fad_full_feedback_fused_candidate_v1'
_KERNEL = None
P1_SHA = '00ffa4cdc27161fb9cfe517ebbef844271dfc6d51d297a264037c2d199d5cea5'
FULL_SHA = '7e6f3f01b6a67d3cf58fdf4e67b0f2e08e0e4184417e464b012d324cc508f0c4'
DYNAMIC = (['W'+str(i) for i in range(5)] + ['PATH'] + ['T'+str(i) for i in range(9)] +
           ['S'+str(i) for i in range(8)] + ['A', 'REMAIN', 'HOST_VALID'])


def _require(ok, message):
    if not ok: raise ValueError(message)


@lru_cache(maxsize=1)
def source_receipt():
    from . import compact_stop_tensor as p1, full_feedback_tensor as full
    for module, expected in ((p1, P1_SHA), (full, FULL_SHA)):
        _require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected,
            'frozen feedback reference source differs')
    return dict(P1_sha256=P1_SHA, full_sha256=FULL_SHA, upstream=p1.source_receipt())


def encode_host_rows(parents, depths):
    _require(len(parents) == len(depths) and 1 <= len(parents) <= 30, 'host row count')
    words = [0]*5
    for i, (parent, depth) in enumerate(zip(parents, depths)):
        _require(type(parent) is int and type(depth) is int and 0 <= parent < 32 and 0 <= depth < 32,
            'host parent/depth range')
        words[i//6] |= (parent | (depth << 5)) << (10*(i%6))
    return tuple(words)


def decode_host_rows(words, count):
    """CPU-only test/diagnostic inverse; not called in the CUDA hot path."""
    _require(len(words) == 5 and type(count) is int and 1 <= count <= 30, 'encoded rows')
    return ([(words[i//6] >> (10*(i%6))) & 31 for i in range(count)],
            [(words[i//6] >> (10*(i%6)+5)) & 31 for i in range(count)])


@dataclass(frozen=True)
class HostPlan:
    depth: int
    nodes: int
    host_valid: bool
    words: tuple
    path_bits: int
    proposed: tuple
    stops: tuple
    accepted: int
    remaining: int

    def scalar_arguments(self):
        return (*self.words, self.path_bits, *self.proposed, *self.stops,
                self.accepted, self.remaining, int(self.host_valid))

    def receipt(self):
        return dict(schema=SCHEMA, depth=self.depth, pool_nodes=8+64*(self.depth-1),
            tree_nodes=self.nodes, custom_kernel_launches=1, packet_bytes=160,
            baseline_root_bytes=8, additional_logical_D2H_bytes=152,
            host_scalar_arguments=26, logical_host_integer_payload_upper_bound_bytes=26*8,
            parameter_payload_scope='scalar values only; excludes pointer arguments, driver ABI padding and JIT metadata; not an explicit H2D copy count',
            full_host_graph_valid=self.host_valid, CPU_host_structure_scan=True,
            runtime_value_specialization=False, actual_compilation_cache_verified=False,
            actual_CUDA_launches=None, actual_DRAM_bytes=None, actual_SRAM_bytes=None,
            conditional_counterfactuals_are_actual_measurements=False,
            new_Target_calls=0, new_argmax_calls=0, new_host_device_value_reads=0,
            allocation_scope='optional independent 160B out; CUDA compiler spills/workspace unmeasured')


def prepare_metadata(result, *, next_root, best, accepted_draft, submitted_ids,
                     remaining_tokens, request_key, state_key, stop_ids):
    import torch
    from . import compact_stop_tensor as p1
    source_receipt()
    _require(type(request_key) is str and request_key and type(state_key) is str and state_key,
        'request/state binding required')
    _require(type(best) is int and best >= 0 and type(accepted_draft) is int and 0 <= accepted_draft <= 8,
        'posterior indices')
    _require(type(remaining_tokens) is int and 1 <= remaining_tokens <= 512, 'remaining tokens')
    _require(type(submitted_ids) in (list, tuple) and len(submitted_ids) == accepted_draft+1
        and all(type(v) is int and 0 <= v < 2**63 for v in submitted_ids), 'unclipped host submitted IDs')
    _require(type(stop_ids) in (set, list, tuple) and 1 <= len(stop_ids) <= 8
        and all(type(v) is int and 0 <= v < 2**63 for v in stop_ids), '1..8 host stop IDs')
    _require(len(submitted_ids) < remaining_tokens and not any(v in stop_ids for v in submitted_ids),
        'clip-return must bypass feedback construction')
    s = result.search; s.validate(result.tree)
    _require(type(s.actual_depth) is int and 1 <= s.actual_depth <= 8, 'depth 1..8')
    token, retrieve, mask, positions = result.tree; m = s.selected.numel()+1
    _require(token.dtype == positions.dtype == torch.int64 and token.shape == (1, m)
        and positions.shape == (m,) and token.device == positions.device == s.scores.device,
        'tree device/dtype/shape')
    _require(retrieve.device.type == mask.device.type == 'cpu' and retrieve.dtype == torch.int64
        and mask.dtype == torch.float32 and retrieve.ndim == 2 and mask.shape == (1, 1, m, m),
        'original CPU retrieve/mask required')
    _require(all(t.is_contiguous() for _, t in s.named_tensors()) and token.is_contiguous()
        and positions.is_contiguous(), 'candidate supports contiguous device operands only')
    _require(best < retrieve.shape[0] and accepted_draft < retrieve.shape[1], 'posterior extent')
    raw = retrieve[best, :accepted_draft+1].tolist()
    _require(raw and raw[0] == 0 and all(0 <= i < m for i in raw), 'accepted path indices')
    _require(isinstance(next_root, torch.Tensor) and next_root.dtype == torch.int64 and next_root.ndim == 0
        and next_root.device == s.scores.device and not next_root.requires_grad, 'original nextroot device scalar')
    valid, parents, depths = p1._host_structure(mask, retrieve, m)
    # _host_structure returns safe placeholders even for malformed host values.
    words = encode_host_rows(parents, [max(0, d) for d in depths])
    return HostPlan(s.actual_depth, m, bool(valid), words,
        sum(v << (5*i) for i, v in enumerate(raw)), tuple(submitted_ids)+(-1,)*(9-len(raw)),
        tuple(sorted(stop_ids))+(-1,)*(8-len(stop_ids)), accepted_draft, remaining_tokens)


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None: return _KERNEL
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError('Triton fused feedback unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=DYNAMIC)
    def fused(SCORES,TOKENS,SELECTED,FRONT,TREE,POS,ROOT,OUT,
              P0,P1,P2,P3,P4,P5,P6,P7,
              W0,W1,W2,W3,W4,PATH,T0,T1,T2,T3,T4,T5,T6,T7,T8,
              S0,S1,S2,S3,S4,S5,S6,S7,A,REMAIN,HOST_VALID,
              D:tl.constexpr,BLOCK:tl.constexpr):
        N:tl.constexpr = 8+64*(D-1)
        M:tl.constexpr = 9 if D==1 else 30
        x=tl.arange(0,BLOCK); active=x<N; ids=x+1
        scores=tl.load(SCORES+x,active,other=-float('inf'))
        tokens=tl.load(TOKENS+x,active,other=0)
        level=tl.where(x<8,1,2+(x-8)//64)
        parent=tl.full((BLOCK,),0,tl.int64)
        pointers=(P0,P1,P2,P3,P4,P5,P6,P7)
        for d in tl.static_range(1,D+1):
            pi=tl.minimum(tl.maximum((x-(0 if d==1 else 8+64*(d-2)))//8,0),7)
            pv=tl.load(pointers[d-1]+pi,active&(level==d),other=0)
            parent=tl.where(level==d,pv,parent)
        safe_parent=tl.minimum(tl.maximum(parent,0),N)
        plevel=tl.where(safe_parent==0,0,tl.where(safe_parent<=8,1,2+(safe_parent-9)//64))
        parent_score=tl.load(SCORES+tl.minimum(tl.maximum(parent-1,0),N-1),active,other=0)
        valid=tl.sum((active&((scores!=scores)|(tl.abs(scores)==float('inf'))|(scores>0)|
            (tokens<0)|(parent<0)|(parent>=ids)|(plevel+1!=level)|
            ((parent!=0)&(scores>parent_score)))).to(tl.int32),0)==0
        valid=valid&(tl.load(P0)==0)
        # Full eight-child groups; sorting checks uniqueness without NxN pairs.
        sorted_tokens=tl.sort(tl.reshape(tokens,(BLOCK//8,8)),descending=False,dim=1)
        previous=tl.gather(sorted_tokens,tl.broadcast_to(tl.maximum(tl.arange(0,8)-1,0)[None,:],(BLOCK//8,8)),axis=1)
        repeated=(sorted_tokens==previous)&(tl.arange(0,8)[None,:]>0)&(tl.arange(0,BLOCK//8)[:,None]*8<N)
        valid=valid&(tl.sum(tl.sum(repeated.to(tl.int32),1),0)==0)

        j=tl.arange(0,32); sel=tl.load(SELECTED+j,j<M-1,other=-1)
        prev=tl.load(SELECTED+tl.maximum(j-1,0),j<M-1,other=-1)
        valid=valid&(tl.sum(((j<M-1)&((sel<0)|(sel>=N)|((j>0)&(sel<=prev)))).to(tl.int32),0)==0)
        safe_sel=tl.minimum(tl.maximum(sel,0),N-1)
        selected_mask=tl.sum(((x[:,None]==safe_sel[None,:])&(j[None,:]<M-1)).to(tl.int32),1)>0
        chosen_scores=tl.load(SCORES+safe_sel,j<M-1,other=float('inf'))
        valid=valid&(tl.min(chosen_scores,0)>=tl.max(tl.where(active&~selected_mask,scores,-float('inf')),0))
        ti=tl.load(SELECTED+tl.maximum(j-1,0),(j>0)&(j<M),other=-1)+1
        tree_ids=tl.where(j==0,0,ti)
        safe_tid=tl.minimum(tl.maximum(tree_ids,0),N)
        actual_tokens=tl.load(TREE+j,j<M,other=0)
        mapped_token=tl.load(TOKENS+tl.minimum(tl.maximum(tree_ids-1,0),N-1),j<M,other=0)
        valid=valid&(tl.sum(((j>0)&(j<M)&(actual_tokens!=mapped_token)).to(tl.int32),0)==0)
        word=tl.where(j//6==0,W0,tl.where(j//6==1,W1,tl.where(j//6==2,W2,tl.where(j//6==3,W3,W4)))).to(tl.uint64)
        hparent=((word>>((j%6)*10))&31).to(tl.int32)
        hdepth=((word>>((j%6)*10+5))&31).to(tl.int32)
        parent_tid=tl.gather(tree_ids,tl.minimum(hparent,M-1),0)
        actual_parent=tl.load(P0) + tl.full((32,),0,tl.int64)
        for d in tl.static_range(1,D+1):
            dep=tl.where(safe_tid<=8,1,2+(safe_tid-9)//64)
            pi=tl.minimum(tl.maximum((safe_tid-1-(0 if d==1 else 8+64*(d-2)))//8,0),7)
            pv=tl.load(pointers[d-1]+pi,(j>0)&(j<M)&(dep==d),other=0)
            actual_parent=tl.where((j>0)&(dep==d),pv,actual_parent)
        actual_pos=tl.load(POS+j,j<M,other=0)
        valid=valid&(HOST_VALID!=0)&(tl.sum(((j<M)&((actual_parent!=parent_tid)|(actual_pos!=hdepth))).to(tl.int32),0)==0)

        step=tl.arange(0,16); g=A+1
        raw=((PATH.to(tl.uint64)>>(tl.minimum(step,8)*5))&31).to(tl.int32)
        path=tl.gather(tree_ids,tl.minimum(raw,M-1),0)
        proposed=tl.where(step==0,T0,tl.where(step==1,T1,tl.where(step==2,T2,tl.where(step==3,T3,
            tl.where(step==4,T4,tl.where(step==5,T5,tl.where(step==6,T6,tl.where(step==7,T7,T8))))))))
        proposed_ok=tl.sum(((step<g)&(tl.gather(actual_tokens,tl.minimum(raw,M-1),0)!=proposed)).to(tl.int32),0)==0
        u=tl.sum(tl.where(step==A,path,0),0)
        root=tl.load(ROOT); valid=valid&proposed_ok&(root>=0)
        safe_u=tl.minimum(tl.maximum(u,0),N)
        uscore=tl.load(SCORES+tl.minimum(tl.maximum(safe_u-1,0),N-1))
        ulevel=tl.where(safe_u==0,0,tl.where(safe_u<=8,1,2+(safe_u-9)//64))
        children=active&(parent==u); correct=children&(tokens==root)
        child_count=tl.sum(children.to(tl.int32),0); correct_count=tl.sum(correct.to(tl.int32),0)
        correct_node=tl.max(tl.where(correct,ids,0),0)
        cs=tl.load(SCORES+tl.minimum(tl.maximum(correct_node-1,0),N-1))
        strict_global=tl.sum((active&(scores>cs)).to(tl.int32),0)
        correct_verified=tl.sum(((j<M)&(tree_ids==correct_node)).to(tl.int32),0)>0
        f=tl.arange(0,8); strict_beam=tl.full((),0,tl.int32)
        final_front=tl.load(FRONT+f)+(0 if D==1 else 8+64*(D-2))+1
        is_frontier=tl.sum((final_front==u).to(tl.int32),0)>0
        for d in tl.static_range(1,D+1):
            start=0 if d==1 else 8+64*(d-2)
            count=8 if d==1 else 64
            if d==D: front=final_front
            else: front=tl.load(pointers[d]+f)
            fi=front-start-1
            fs=tl.load(SCORES+start+tl.minimum(tl.maximum(fi,0),count-1))
            front_sorted=tl.sort(front,descending=False)
            front_prev=tl.gather(front_sorted,tl.maximum(f-1,0),0)
            valid=valid&(tl.sum(((fi<0)|(fi>=count)|((f>0)&(front_sorted==front_prev))).to(tl.int32),0)==0)
            fc=tl.sum((ids[:,None]==front[None,:]).to(tl.int32),1)>0
            mx=tl.max(tl.where(active&(x>=start)&(x<start+count)&~fc,scores,-float('inf')),0)
            valid=valid&(tl.min(fs,0)>=mx)
            strict_beam+=tl.where(ulevel==d,tl.sum((fs>uscore).to(tl.int32),0),0)
        valid=valid&(tl.sum(tl.sum((active[:,None]&(parent[:,None]==final_front[None,:])).to(tl.int32),1),0)==0)

        expanded=child_count>0; complete=expanded&(child_count==8)&(correct_count<=1)
        missing=complete&(correct_count==0)
        pruned=complete&(correct_count==1)&~correct_verified&(strict_global>=29)
        eliminated=~expanded&~is_frontier&(child_count==0)&(strict_beam==8)
        frontier_unknown=~expanded&is_frontier
        kind=tl.where(frontier_unknown,4,tl.where(eliminated,3,tl.where(pruned,2,tl.where(missing,1,0))))
        witness=tl.where(complete&(correct_count==1),correct_node,u)
        kind=tl.where(valid,kind,0);witness=tl.where(valid,witness,-1)
        flags=tl.where(valid,7,1)|tl.where(valid&(pruned|eliminated),8,0)|tl.where(kind==0,128,0)
        eos=(root==S0)|(root==S1)|(root==S2)|(root==S3)|(root==S4)|(root==S5)|(root==S6)|(root==S7)
        terminal=(eos|(g+1==REMAIN))&proposed_ok&(root>=0)
        kind=tl.where(terminal,5,kind);witness=tl.where(terminal,-1,witness)
        flags=tl.where(terminal,tl.where(eos,49,81),flags)
        tl.store(OUT,root);tl.store(OUT+1,kind);tl.store(OUT+2,witness);tl.store(OUT+3,flags)
        qualified=valid&~terminal&(kind>=1)&(kind<=4)
        failure=(kind==1)|(kind==2)|(kind==3)
        for d in tl.static_range(1,9):
            code=tl.full((),0,tl.int32);gain=tl.full((),-1,tl.int64)
            if d>D:
                eligible=qualified&failure
                code=tl.where(eligible,2,0);gain=tl.where(eligible,g,-1)
            elif d<D:
                n=8+64*(d-1)
                b=8 if d==1 else 29
                ranked=tl.sort(tl.where(x<n,scores,-float('inf')),descending=True)
                cutoff=tl.sum(tl.where(x==b-1,ranked,0.),0)
                if d==1: strict=tl.full((),True,tl.int1)
                else: strict=cutoff>tl.sum(tl.where(x==b,ranked,0.),0)
                chosen=(x<n)&(scores>=cutoff)
                parent_chosen=tl.gather(chosen,tl.minimum(tl.maximum(parent-1,0),BLOCK-1),0)
                closed=tl.sum((chosen&(parent!=0)&~parent_chosen).to(tl.int32),0)==0
                present=tl.gather(chosen,tl.minimum(tl.maximum(path-1,0),BLOCK-1),0)
                first_missing=tl.min(tl.where((step>0)&(step<g)&~present,step,g),0)
                lost=first_missing<g
                correct_visible=tl.sum((chosen&(parent==u)&(tokens==root)).to(tl.int32),0)>0
                proposed_gain=tl.where(lost,first_missing,g)
                eligible=qualified&strict&closed&(lost|~correct_visible)&(proposed_gain>=1)&(proposed_gain<=d+1)
                code=tl.where(eligible,1,0);gain=tl.where(eligible,proposed_gain,-1)
            tl.store(OUT+3+d,code);tl.store(OUT+11+d,gain)
    _KERNEL=fused
    return _KERNEL


def cpu_threshold_reference(result, **kwargs):
    """Explicit CPU oracle for the sort-threshold rewrite; never a fallback.

    Local4 uses the frozen complete P1 reference. This validates the changed
    shallow membership algorithm, not execution of the Triton graph checks.
    """
    import torch
    from . import compact_search_feedback as ref, compact_stop_tensor as p1
    _require(all(t.device.type=='cpu' for _,t in result.search.named_tensors())
        and kwargs['next_root'].device.type=='cpu','CPU threshold reference only')
    plan=prepare_metadata(result,**kwargs)
    base=p1.classify_stop_tensor(result,proposed_ids=kwargs['submitted_ids'],**kwargs).packet.tolist()
    output=base+[0]*8+[-1]*8
    if base[1] not in (1,2,3,4) or base[3]&144: return torch.tensor(output,dtype=torch.int64)
    graph=ref._cpu_graph(result)
    path=[graph['selected'][i] for i in graph['rows'][kwargs['best']][:plan.accepted+1]]
    root=int(kwargs['next_root'])
    for d in range(1,9):
        if d==plan.depth:continue
        if d>plan.depth:
            if base[1] in (1,2,3):output[3+d],output[11+d]=2,len(path)
            continue
        n=8+64*(d-1); b=min(29,n); scores=graph['scores'][:n]; ranked=sorted(scores,reverse=True)
        strict=n==b or ranked[b-1]>ranked[b]
        chosen={i+1 for i,s in enumerate(scores) if s>=ranked[b-1]}
        closed=all(graph['parents'][i]==0 or graph['parents'][i] in chosen for i in chosen)
        first_missing=next((i for i,node in enumerate(path) if i and node not in chosen),len(path))
        correct_visible=any(graph['parents'][i]==path[-1] and graph['tokens'][i-1]==root for i in chosen)
        gain=first_missing
        if strict and closed and (gain<len(path) or not correct_visible) and 1<=gain<=d+1:
            output[3+d],output[11+d]=1,gain
    return torch.tensor(output,dtype=torch.int64)


def build_feedback_packet(result, *, next_root, best, accepted_draft, submitted_ids,
                          remaining_tokens, request_key, state_key, stop_ids, out=None):
    import torch
    plan=prepare_metadata(result,next_root=next_root,best=best,accepted_draft=accepted_draft,
        submitted_ids=submitted_ids,remaining_tokens=remaining_tokens,request_key=request_key,
        state_key=state_key,stop_ids=stop_ids)
    s=result.search
    _require(s.scores.device.type=='cuda', 'fused candidate requires CUDA; no CPU/Torch fallback')
    if out is None: out=torch.empty((20,),dtype=torch.int64,device=s.scores.device)
    _require(isinstance(out,torch.Tensor) and out.dtype==torch.int64 and out.shape==(20,)
        and out.is_contiguous() and out.device==s.scores.device and not out.requires_grad,'independent device int64[20] out')
    sid=lambda t:(str(t.device),t.untyped_storage().data_ptr())
    inputs=[t for _,t in s.named_tensors()]+list(result.tree)+[next_root]
    _require(sid(out) not in {sid(t) for t in inputs},'out aliases input backing')
    pointers=(*s.parents,)+(s.parents[0],)*(8-len(s.parents))
    kernel=_get_kernel()
    kernel[(1,)](s.scores,s.tokens,s.selected,s.frontier,result.tree[0],result.tree[3],next_root,out,
        *pointers,*plan.scalar_arguments(),D=plan.depth,BLOCK=1<<(8+64*(plan.depth-1)-1).bit_length(),
        num_warps=8,num_stages=1,enable_fp_fusion=False)
    return out
