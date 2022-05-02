from logging import getLogger

from continual_transformers.co_re_mha import CoReMultiheadAttention
from continual_transformers.co_si_mha import CoSiMultiheadAttention

logger = getLogger(__name__)


def _register_ptflops():
    try:
        import ptflops

        if hasattr(ptflops, "pytorch_ops"):  # >= v0.6.8
            fc = ptflops.pytorch_ops
        else:  # < v0.6.7
            fc = ptflops.flops_counter

        def get_hook(Module):
            def hook(module, input, output):
                module.__flops__ += Module.flops(module)

            return hook

        fc.MODULES_MAPPING[CoReMultiheadAttention] = get_hook(CoReMultiheadAttention)
        fc.MODULES_MAPPING[CoSiMultiheadAttention] = get_hook(CoSiMultiheadAttention)

    except ModuleNotFoundError:  # pragma: no cover
        pass
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to add flops_counter_hook: {e}")


_register_ptflops()
