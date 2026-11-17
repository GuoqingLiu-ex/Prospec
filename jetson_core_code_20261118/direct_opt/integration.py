"""Independent compression/depth arms on the unchanged Target+Draft runner."""
from contextlib import contextmanager
from scheme_context import SCHEMES as OLD_SCHEMES, SchemeRuntime as OldRuntime
from compression import FrozenQueryAdaptor
from compression_graph import CudaGraphAdaptor

SCHEMES = {k: dict(v) for k, v in OLD_SCHEMES.items()}
for name, backend in [('frozen_bmm', 'bmm'), ('frozen_sdpa', 'sdpa')]:
    SCHEMES[name] = dict(depth='fixed0', adaptor=name, compression=None, ratio=1.0)
SCHEMES['frozen_graph'] = dict(depth='fixed0', adaptor='frozen_graph', compression=None, ratio=1.0)
SCHEMES['reorder_skip_fixed2'] = dict(depth='fixed2', adaptor='model', compression=None, ratio=1.0)
SCHEMES['reorder_direct'] = dict(depth='direct', adaptor='model', compression=None, ratio=1.0)
SCHEMES['native_direct'] = dict(depth='direct', adaptor='native', compression=None, ratio=1.0)
SCHEMES['frozen_direct'] = dict(depth='direct', adaptor='frozen_bmm', compression=None, ratio=1.0)
SCHEMES['reorder_direct_uniform'] = dict(SCHEMES['reorder_direct'], depth_allocation='uniform',
                                          depth_allocation_seed=20261002)
for seed in (20261002, 20261003, 20261004):
    SCHEMES[f'reorder_direct_random_{seed}'] = dict(
        SCHEMES['reorder_direct'], depth_allocation='random', depth_allocation_seed=seed)


class SchemeRuntime(OldRuntime):
    def __init__(self, draft, **kwargs):
        super().__init__(draft, **kwargs)
        self.frozen = {name: FrozenQueryAdaptor(self.adaptor, backend=backend)
                       for name, backend in [('frozen_bmm', 'bmm'), ('frozen_sdpa', 'sdpa')]}

        self.graph = CudaGraphAdaptor(self.frozen['frozen_bmm'], max_entries=16)

    @contextmanager
    def activate(self, joint, scheme):
        name = scheme['adaptor']
        if name not in self.frozen and name != 'frozen_graph':
            with super().activate(joint, scheme) as state:
                yield state
            return
        base = dict(scheme, adaptor='native')
        with super().activate(joint, base) as state:
            state['adaptor'] = name
            if name == 'frozen_graph':
                state['graph_setup'] = []
                def forward(x):
                    sizes = state['group_sizes']
                    if self.graph._key(x, sizes) not in self.graph._entries:
                        state['graph_setup'].append(self.graph.prepare_for(x, sizes))
                    return self.graph(x, group_sizes=sizes)
                self.adaptor.forward = forward
            else:
                state['preparation'] = self.frozen[name].preparation_metadata()
                self.adaptor.forward = lambda x: self.frozen[name](x, group_sizes=state['group_sizes'])
            yield state
