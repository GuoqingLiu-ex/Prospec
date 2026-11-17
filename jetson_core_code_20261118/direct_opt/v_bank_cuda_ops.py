"""Isolated CUDA visual-V pack/fill primitives; no model, cache or fallback.

Triton is imported lazily. CPU tests exercise metadata, launch plans, and source
contracts only; they do not execute these kernels or establish GPU numerics.
The caller supplies actual visual positions through a CPU-validated partition,
proves K/source-V storage independence, and retains exclusive immutable ownership.

Pack returns (fixed_visual_bank, compact_prompt_text_V), separately. A setup
validate_pack_status() boundary must succeed before any fill. Fill performs no
host tensor read/synchronization: device-invalid inputs produce NaN operands and
set a persistent error flag. check_status() at an explicit boundary is mandatory
before accepting any output/result. The flag is never reset by this API. Device
flags do not replace CUDA launch/runtime error handling or source provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch
from partitioned_v_bank import VisualPartition

D, HEADS, GROUP, TILE = 128, 4, 32, 8
MODES = ('fp16_partition', 'int8_visual')
PACK_DYNAMIC = ('P','M','R','SH','SN','SD')
FILL_DYNAMIC = ('P','M','R','S','NF','TH','TN','TD')
_KERNELS = None


def _bytes(x):return x.numel()*x.element_size()
def _sid(x):return str(x.device),x.untyped_storage().data_ptr()


def _positive(value,name,minimum=1):
    if type(value) is not int or value<minimum or value>2**31-1:
        raise ValueError(name+' must be a bounded positive host integer')
    return value


def _construct(cls,**fields):
    obj=object.__new__(cls)
    for k,v in fields.items():object.__setattr__(obj,k,v)
    return obj


@dataclass(frozen=True,eq=False,init=False)
class PartitionRoute:
    partition: VisualPartition
    prefix_route: torch.Tensor = field(repr=False)
    _version: int

    def __init__(self,*a,**kw):raise TypeError('use prepare_partition_route')
    @property
    def P(self):return self.partition.prefix_tokens
    @property
    def M(self):return len(self.partition.visual)
    @property
    def R(self):return len(self.partition.nonvisual)

    def validate_metadata(self):
        self.partition.validate_metadata()
        route=self.prefix_route
        if (route.dtype!=torch.int32 or route.shape!=(self.P,) or not route.is_contiguous()
                or route.device.type not in ('cpu','cuda') or route._version!=self._version):
            raise ValueError('shared device route metadata/mutation mismatch')

    def receipt(self):
        self.validate_metadata()
        return dict(P=self.P,M=self.M,R=self.R,device=str(self.prefix_route.device),
            device_route_backing_bytes=self.prefix_route.untyped_storage().nbytes(),
            route_uploaded_once_per_request=True,host_partition=self.partition.receipt(),
            upload_logical_read_bytes=4*self.P,upload_logical_write_bytes=4*self.P,
            scope='shared route allocated once; host/device setup costs and transfer are not pack/fill costs')


def prepare_partition_route(partition,device):
    """One explicit route copy; CPU device is for metadata tests, never a GPU fallback."""
    if not isinstance(partition,VisualPartition):raise TypeError('CPU-validated VisualPartition required')
    partition.validate_metadata();_positive(partition.prefix_tokens,'P')
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):raise ValueError('CPU metadata or CUDA route required')
    with torch.inference_mode(False):
        route=partition.prefix_route.to(device=device,copy=True)
    return _construct(PartitionRoute,partition=partition,prefix_route=route,_version=route._version)


def pack_launch_plan(P,M,mode):
    _positive(P,'P');_positive(M,'M')
    if M>P or mode not in MODES:raise ValueError('invalid partition count/mode')
    return dict(grid=(HEADS*P,),num_warps=4,num_stages=1,
        constexpr={'QUANT':mode=='int8_visual'},runtime_scalars=PACK_DYNAMIC,
        token_padding=0,source_elements_read=HEADS*P*D,
        source_payload_read_bytes=2*HEADS*P*D,
        output_payload_write_bytes=(HEADS*M*D*(1 if mode=='int8_visual' else 2)
            +(HEADS*M*(D//GROUP)*4 if mode=='int8_visual' else 0)+2*HEADS*(P-M)*D),
        route_logical_reads_bytes=4*HEADS*P,flag_initialization_bytes=4,
        flag_atomic_traffic_not_counted=True,kernel_launches=1,
        kernel_launch_scope='custom_pack_kernel_only',total_device_launches_measured=False,
        additional_costs_unmeasured=(
            'shared route upload (once per request, outside pack)',
            'torch.zeros error flag initialization',
            'explicit check_status torch.stack/ne/any/item reduction and host synchronization'),
        quantization_rule='FP32 div_rn(maxabs,127), zero scale1, div_rn(value,scale), nearbyint ties-to-even, clamp[-127,127]',
        explicit_tile_value_elements=128,tile_residency_measured=False)


def fill_launch_plan(P,M,S,mode):
    _positive(S,'S');base=pack_launch_plan(P,M,mode)
    if S<P:raise ValueError('fill S cannot precede complete prompt')
    groups=math.ceil(S/TILE)
    visual=HEADS*M*D
    return dict(grid=(HEADS,groups),num_warps=4,num_stages=1,
        constexpr={'QUANT':mode=='int8_visual','BLOCK':TILE},runtime_scalars=FILL_DYNAMIC,
        dynamic_length_specialization_disabled=True,actual_compile_count_verified=False,
        ideal_source_once_read_bytes=(visual*(1 if mode=='int8_visual' else 2)
            +(visual//GROUP*4 if mode=='int8_visual' else 0)+2*HEADS*(S-M)*D),
        operand_write_bytes=4*HEADS*S*D,
        route_logical_read_bytes=4*HEADS*P,
        scale_load_lanes=visual if mode=='int8_visual' else 0,
        scale_unique_elements=visual//GROUP if mode=='int8_visual' else 0,
        flag_read_programs=HEADS*groups,flag_atomic_traffic_not_counted=True,
        per_program_tile_elements=TILE*D,kernel_launches=1,
        kernel_launch_scope='custom_fill_kernel_only',total_device_launches_measured=False,
        additional_costs_unmeasured=(
            'host metadata guards and custom kernel launch',
            'explicit check_status torch.stack/ne/any/item at acceptance boundary (outside hot fill)'),
        scope='ideal source-once payload is not DRAM; scale lanes/route/flags may repeatedly read the same data; '
              'tile values are compiler/register intent, not SRAM or measured live workspace')


def _tensor_meta(x,name,shape,dtype,device=None,contiguous=False):
    if (not isinstance(x,torch.Tensor) or x.layout!=torch.strided or tuple(x.shape)!=tuple(shape)
            or x.dtype!=dtype or x.requires_grad or x.device.type not in ('cpu','cuda')
            or (device is not None and x.device!=device)):
        raise ValueError(name+' shape/dtype/device/inference metadata mismatch')
    if contiguous and not x.is_contiguous():raise ValueError(name+' must be contiguous')


def validate_pack_metadata(values,route,mode):
    if not isinstance(route,PartitionRoute):raise TypeError('prepared PartitionRoute required')
    route.validate_metadata()
    if mode not in MODES:raise ValueError('unsupported mode')
    _tensor_meta(values,'source V',(1,HEADS,route.P,D),torch.float16,route.prefix_route.device)
    if values.stride(-1)!=1 or values.stride(-2)!=D or values.stride(1)<route.P*D:
        raise ValueError('native nonoverlapping source V strides required')
    if _sid(values)==_sid(route.prefix_route):raise ValueError('source and route alias')
    return pack_launch_plan(route.P,route.M,mode)


@dataclass(eq=False,init=False)
class CudaVisualVBank:
    route: PartitionRoute
    mode: str
    visual_values: torch.Tensor = field(repr=False)
    scales: torch.Tensor | None = field(repr=False)
    error_flag: torch.Tensor = field(repr=False)
    _versions: tuple
    validated: bool
    failed: bool

    def __init__(self,*a,**kw):raise TypeError('created only by pack_v_cuda')

    def validate_metadata(self):
        self.route.validate_metadata();device=self.route.prefix_route.device
        if self.mode not in MODES:raise ValueError('invalid bank mode')
        _tensor_meta(self.visual_values,'visual',(1,HEADS,self.route.M,D),
            torch.int8 if self.mode=='int8_visual' else torch.float16,device,True)
        if self.mode=='int8_visual':
            _tensor_meta(self.scales,'scales',(1,HEADS,self.route.M,D//GROUP),torch.float32,device,True)
        elif self.scales is not None:raise ValueError('FP16 control carries scales')
        _tensor_meta(self.error_flag,'error flag',(),torch.int32,device,True)
        if (self.visual_values._version,None if self.scales is None else self.scales._version)!=self._versions:
            raise ValueError('immutable visual bank/scales changed')
        owned=[self.visual_values,self.route.prefix_route,self.error_flag]+([] if self.scales is None else [self.scales])
        if len({_sid(x) for x in owned})!=len(owned):raise ValueError('bank/route/flag storage alias')

    def storage_receipt(self):
        self.validate_metadata()
        payload=[self.visual_values,self.error_flag]+([] if self.scales is None else [self.scales])
        return dict(mode=self.mode,visual_tokens=self.route.M,visual_bytes=_bytes(self.visual_values),
            scale_bytes=0 if self.scales is None else _bytes(self.scales),error_flag_bytes=4,
            unique_bank_backing_bytes=sum(x.untyped_storage().nbytes() for x in payload),
            shared_route=self.route.receipt(),shared_route_excluded_from_bank_total=True,
            text_bank_not_retained=True,source_V_not_retained=True,output_operand_not_retained=True,
            validated_at_explicit_boundary=self.validated,failed=self.failed,actual_DRAM_measured=False,
            GPU_numerics_validated=False,scope='persistent known roots only; excludes caller text/K/operand and allocator overhead')


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:return _KERNELS
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as exc:
        raise RuntimeError('Triton CUDA primitives unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=PACK_DYNAMIC)
    def _pack(SRC,ROUTE,VIS,SCALES,TEXT,ERROR,P,M,R,SH,SN,SD,QUANT:tl.constexpr):
        row=tl.program_id(0)
        j=row%P;h=row//P
        route=tl.load(ROUTE+j)
        ri=route.to(tl.int64)
        visual=(ri>=0)&(ri<M);text=(ri<0)&(-ri-1<R)
        valid_route=visual|text
        d=tl.arange(0,128)
        source=tl.load(SRC+h.to(tl.int64)*SH+j.to(tl.int64)*SN+d*SD)
        x=source.to(tl.float32)
        finite=(x==x)&(tl.abs(x)!=float('inf'))
        if not valid_route:tl.atomic_or(ERROR,1)
        if tl.sum((~finite).to(tl.int32),0)>0:tl.atomic_or(ERROR,2)
        if visual:
            if QUANT:
                safe=tl.where(finite,x,0.)
                grouped=tl.reshape(safe,(4,32))
                maximum=tl.max(tl.abs(grouped),1)
                scale=tl.where(maximum==0,1.,tl.div_rn(maximum,127.))
                codes=libdevice.nearbyint(tl.div_rn(grouped,scale[:,None]))
                codes=tl.minimum(tl.maximum(codes,-127.),127.).to(tl.int8)
                tl.store(VIS+(h.to(tl.int64)*M+ri)*128+d,tl.reshape(codes,(128,)))
                tl.store(SCALES+(h.to(tl.int64)*M+ri)*4+tl.arange(0,4),scale)
            else:
                # Original FP16 bits, including signed zero; no value arithmetic.
                tl.store(VIS+(h.to(tl.int64)*M+ri)*128+d,source)
        if text:
            tl.store(TEXT+(h.to(tl.int64)*R-ri-1)*128+d,source)

    @triton.jit(do_not_specialize=FILL_DYNAMIC)
    def _fill(VIS,SCALES,TEXT,ROUTE,ERROR,OUT,P,M,R,S,NF,TH,TN,TD,
              QUANT:tl.constexpr,BLOCK:tl.constexpr):
        h=tl.program_id(0)
        j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        d=tl.arange(0,128)
        inside=j<S
        prefix=inside&(j<P)
        route=tl.load(ROUTE+j,mask=prefix,other=-1).to(tl.int64)
        visual=prefix&(route>=0)&(route<M)
        ti=tl.where(prefix,-route-1,R+j-P).to(tl.int64)
        # Prefix routes may only address the compact prompt bank. A malformed
        # negative route must not become valid when generated text grows NF.
        prefix_text=prefix&(route<0)&(ti>=0)&(ti<R)
        generated_text=inside&(~prefix)&(ti>=R)&(ti<NF)
        text=prefix_text|generated_text
        valid=visual|text
        bad_route=inside&(~valid)
        vis=tl.load(VIS+(h.to(tl.int64)*M+route[:,None])*128+d[None,:],
                    mask=visual[:,None],other=0).to(tl.float32)
        if QUANT:
            scale=tl.load(SCALES+(h.to(tl.int64)*M+route[:,None])*4+d[None,:]//32,
                          mask=visual[:,None],other=1.)
            scale_ok=(scale==scale)&(tl.abs(scale)!=float('inf'))&(scale>0)
            vis=vis*scale
        else:
            scale_ok=tl.full((BLOCK,128),True,tl.int1)
        text_value=tl.load(TEXT+h.to(tl.int64)*TH+ti[:,None]*TN+d[None,:]*TD,
                           mask=text[:,None],other=0).to(tl.float32)
        value=tl.where(visual[:,None],vis,text_value)
        finite=(value==value)&(tl.abs(value)!=float('inf'))
        bad_element=inside[:,None]&((~finite)|(~scale_ok)|bad_route[:,None])
        if tl.sum(tl.sum(bad_element.to(tl.int32),1),0)>0:tl.atomic_or(ERROR,4)
        # A persistent earlier error poisons later fills as well. A same-launch
        # failing CTA poisons its own lanes; the mandatory explicit status check
        # rejects the whole result even if other CTAs raced before its atomic.
        prior=tl.load(ERROR)
        value=tl.where(bad_element|(prior!=0),float('nan'),value)
        tl.store(OUT+(h.to(tl.int64)*S+j[:,None])*128+d[None,:],value,mask=inside[:,None])

    _KERNELS=(_pack,_fill)
    return _KERNELS


def _require_cuda(tensor):
    if tensor.device.type!='cuda':raise ValueError('CUDA execution required; no CPU fallback')


@torch.no_grad()
def pack_v_cuda(values,route,*,mode='fp16_partition'):
    plan=validate_pack_metadata(values,route,mode);_require_cuda(values)
    pack,_=_get_kernels()
    with torch.inference_mode(False):
        visual=torch.empty((1,HEADS,route.M,D),device=values.device,
            dtype=torch.int8 if mode=='int8_visual' else torch.float16)
        scales=torch.empty((1,HEADS,route.M,D//GROUP),device=values.device,dtype=torch.float32) if mode=='int8_visual' else None
        text=torch.empty((1,HEADS,route.R,D),device=values.device,dtype=torch.float16)
        flag=torch.zeros((),device=values.device,dtype=torch.int32)
    outputs=[visual,text,flag]+([] if scales is None else [scales])
    if any(x.numel() and _sid(x) in (_sid(values),_sid(route.prefix_route)) for x in outputs):
        raise RuntimeError('pack output aliases input')
    pack[plan['grid']](values,route.prefix_route,visual,scales if scales is not None else visual,text,flag,
        route.P,route.M,route.R,*values.stride()[1:],QUANT=mode=='int8_visual',num_warps=4,num_stages=1,
        enable_fp_fusion=False)
    bank=_construct(CudaVisualVBank,route=route,mode=mode,visual_values=visual,scales=scales,error_flag=flag,
        _versions=(visual._version,None if scales is None else scales._version),validated=False,failed=False)
    bank.validate_metadata()
    return bank,text


def check_status(banks,*,stage):
    """Explicit synchronization/readback boundary; never invoked from fill.

    A caller must call this after diagnostic/request work before accepting it.
    Reading an error poisons all supplied banks; no status flag is ever cleared.
    """
    banks=tuple(banks)
    if not banks or not isinstance(stage,str) or not stage:raise ValueError('nonempty banks and explicit stage required')
    for bank in banks:bank.validate_metadata()
    if any(bank.error_flag.device!=banks[0].error_flag.device for bank in banks):
        raise ValueError('status flags must share a device')
    if any(bank.failed for bank in banks):raise RuntimeError('previous failure; banks cannot be reused')
    try:
        combined=torch.stack([bank.error_flag for bank in banks]).ne(0).any()
        invalid=bool(combined.item())  # one explicit aggregate host read
    except BaseException:
        for bank in banks:bank.failed=True;bank.validated=False
        raise
    if invalid:
        for bank in banks:bank.failed=True;bank.validated=False
        raise RuntimeError('visual V device error at '+stage+'; output/result must not be accepted')
    return dict(stage=stage,banks=len(banks),explicit_host_read=True,status_valid=True,
                operations=('torch.stack','Tensor.ne','Tensor.any','Tensor.item'),
                actual_device_launches_measured=False,
                synchronization_and_temporary_reduction_cost_excluded_from_kernel_receipts=True)


def validate_pack_status(banks):
    banks=tuple(banks)
    receipt=check_status(banks,stage='after_pack_before_fill')
    for bank in banks:bank.validated=True
    return receipt


def validate_fill_metadata(out,bank,text_values):
    if not isinstance(bank,CudaVisualVBank):raise TypeError('CudaVisualVBank required')
    bank.validate_metadata()
    if bank.failed or not bank.validated:raise RuntimeError('pack status must be explicitly validated before fill')
    nf=text_values.shape[2] if isinstance(text_values,torch.Tensor) and text_values.ndim==4 else -1
    S=bank.route.P+nf-bank.route.R
    _positive(S,'S')
    if S<bank.route.P:raise ValueError('fill S cannot precede complete prompt')
    device=bank.visual_values.device
    _tensor_meta(text_values,'current text V',(1,HEADS,nf,D),torch.float16,device)
    if text_values.stride(-1)!=1 or text_values.stride(-2)!=D or text_values.stride(1)<nf*D:
        raise ValueError('nonoverlapping text rows/heads required')
    _tensor_meta(out,'FP32 operand',(1,HEADS,S,D),torch.float32,device,True)
    sources=[bank.route.prefix_route,bank.visual_values,bank.error_flag,text_values]+([] if bank.scales is None else [bank.scales])
    if any(x.numel() and _sid(out)==_sid(x) for x in sources):raise ValueError('out aliases route/flag/visual/scales/text')
    return S


@torch.no_grad()
def fill_value_fp32_(out,bank,text_values):
    S=validate_fill_metadata(out,bank,text_values);_require_cuda(out)
    _,fill=_get_kernels()
    try:
        fill[(HEADS,(S+TILE-1)//TILE)](bank.visual_values,bank.scales if bank.scales is not None else bank.visual_values,
            text_values,bank.route.prefix_route,bank.error_flag,out,bank.route.P,bank.route.M,bank.route.R,
            out.shape[2],text_values.shape[2],*text_values.stride()[1:],
            QUANT=bank.mode=='int8_visual',BLOCK=TILE,num_warps=4,num_stages=1,enable_fp_fusion=False)
    except BaseException:
        bank.failed=True;bank.validated=False
        raise
    return out
