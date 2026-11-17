"""Move three frozen constructor AST checks to explicit ModelOps preparation.

Each request still checks source bytes and live method/function identity. Only
the constructor receives independent copies of the genuine cold receipts;
runtime arithmetic, guards, feedback and histories remain inherited from Fast.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
from pathlib import Path
import threading
from types import CodeType, FunctionType

from . import fast_fad_executor as fast
from . import ordinary_depth_round as ordinary
from . import production_depth_executor as production
from . import production_fad_executor as fad
from . import traced_topk as traced
from ..focusdraft import real_system_executor as real

_GUARDS = ((ordinary, 'verify_core_source'), (production, 'verify_production_source'),
           (fad, 'verify_fad_source'))
_CLASSES = ((real, 'RealExecutor'), (ordinary, 'OrdinaryDepthExecutor'),
            (production, 'ProductionDepthExecutor'), (fad, 'ProductionFADExecutor'),
            (fast, 'FastFADExecutor'))
_EXPECTED = dict(fast.EXPECTED,
    **{'fast_fad_executor.py': '30742a881ac3f189e051abd2ea06a4f262fd5f8a288c6818d6c901a64c628ac0',
       'traced_topk.py': 'ff7fe7f36e87d3fd96c2ba2ab29e3ece1ba609ec320140178558054893a63a12'})
_FILES = {Path(fast.__file__).parent / name: digest for name, digest in _EXPECTED.items()}
_FILES[Path(real.__file__)] = ordinary.BASE_EXECUTOR_SHA
_FILES[traced.AUTHOR_PATH] = traced.AUTHOR_SHA256
_LOCK = threading.Lock()
_CACHE = None


def _require(ok, message):
    if not ok:
        raise ValueError('cached setup: ' + message)


def _source_hashes():
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in _FILES}
    _require(all(hashes[str(path)] == expected for path, expected in _FILES.items()), 'source bytes changed')
    return hashes


def _functions(value):
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, property):
        return [fn for fn in (value.fget, value.fset, value.fdel) if fn is not None]
    return [value] if isinstance(value, FunctionType) else []


def _bindings():
    items = [(module, name, getattr(module, name)) for module, name in _GUARDS]
    for module, name in _CLASSES:
        cls = getattr(module, name)
        items.append((module, name, cls))
        items.extend((cls, key, value) for key, value in vars(cls).items() if _functions(value))
    return [(owner, name, value, [(fn, fn.__code__, fn.__wrapped__ if hasattr(fn, '__wrapped__') else None)
            for root in _functions(value) for fn in _unwrap_chain(root)])
            for owner, name, value in items]


def _unwrap_chain(fn):
    chain = []
    while True:
        _require(isinstance(fn, FunctionType) and all(fn is not prior for prior in chain), 'invalid callable wrapper')
        chain.append(fn)
        if not hasattr(fn, '__wrapped__'):
            return chain
        fn = fn.__wrapped__


def _check(cache):
    _require(_source_hashes() == cache['hashes'], 'source changed after preparation')
    current = _bindings()
    _require(len(current) == len(cache['bindings']), 'class methods changed')
    for actual, saved in zip(current, cache['bindings']):
        _require(actual[0] is saved[0] and actual[1] == saved[1] and actual[2] is saved[2]
                 and len(actual[3]) == len(saved[3])
                 and all(a[0] is b[0] and a[1] is b[1] and a[2] is b[2]
                         for a, b in zip(actual[3], saved[3])), 'callable identity/code changed')


def _cold_code_check(bindings):
    """Compile frozen bytes without executing them; reject pre-prepare patches."""
    compiled = {str(p): compile(p.read_bytes(), str(p), 'exec') for p in _FILES}
    def child(code, name):
        return next((c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == name), None)
    for owner, name, value, chains in bindings:
        if not chains:
            continue
        module = inspect.getmodule(owner) if isinstance(owner, type) else owner
        template = compiled[str(Path(module.__file__))]
        if isinstance(owner, type):
            template = child(template, owner.__name__)
        for fn, _, _ in chains:
            if hasattr(fn, '__wrapped__'):
                continue
            _require(template is not None and fn.__code__ == child(template, fn.__code__.co_name),
                     'live callable differs from frozen source: ' + name)


def prepare_cached_setup():
    """Call once during ModelOps setup, before any request/generation timer."""
    global _CACHE
    _require(_LOCK.acquire(blocking=False), 'nested/concurrent preparation or constructor')
    try:
        if _CACHE is None:
            hashes, bindings = _source_hashes(), _bindings()
            _cold_code_check(bindings)
            receipts = [getattr(module, name)() for module, name in _GUARDS]
            # Existing Fast and author lru-cached setup checks remain genuine.
            fast.verify_fast_source()
            traced._setup_source_guard()
            candidate = dict(hashes=hashes, bindings=bindings, receipts=copy.deepcopy(receipts))
            _check(candidate)
            _CACHE = candidate
        else:
            _check(_CACHE)
        return dict(source_sha256=dict(_CACHE['hashes']),
                    guard_receipts=copy.deepcopy(_CACHE['receipts']), constructor_AST_checks_cached=True)
    finally:
        _LOCK.release()


class CachedSetupFastFADExecutor(fast.FastFADExecutor):
    def __init__(self, *args, **kwargs):
        _require(_LOCK.acquire(blocking=False), 'nested/concurrent constructor')
        saved = []
        try:
            _require(_CACHE is not None, 'prepare_cached_setup must run before request construction')
            _check(_CACHE)
            thread_id, calls = threading.get_ident(), [0, 0, 0]
            for index, (module, name) in enumerate(_GUARDS):
                original = getattr(module, name)
                def receipt(index=index):
                    _require(threading.get_ident() == thread_id and calls[index] == 0,
                             'foreign thread or repeated guard call')
                    calls[index] += 1
                    return copy.deepcopy(_CACHE['receipts'][index])
                saved.append((module, name, original, receipt))
                setattr(module, name, receipt)
            super().__init__(*args, **kwargs)
            _require(calls == [1, 1, 1], 'constructor did not consume each frozen guard')
        finally:
            changed = any(getattr(module, name) is not replacement
                          for module, name, original, replacement in saved)
            for module, name, original, replacement in reversed(saved):
                setattr(module, name, original)
            try:
                if saved:
                    _check(_CACHE)
                    _require(not changed, 'temporary constructor guard was replaced')
            finally:
                _LOCK.release()
