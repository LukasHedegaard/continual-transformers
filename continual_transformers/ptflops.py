from logging import getLogger
from continual_transformers.co_re_mha import CoReMultiheadAttention
from continual_transformers.co_si_mha import CoSiMultiheadAttention

logger = getLogger(__name__)


def _register_ptflops():
    try:
        from ptflops import flops_counter as fc

        def get_hook(Module):
            def hook(module, input, output):
                return Module.flops(module)

            return hook

        fc.MODULES_MAPPING[CoReMultiheadAttention] = get_hook(CoReMultiheadAttention)
        fc.MODULES_MAPPING[CoSiMultiheadAttention] = get_hook(CoSiMultiheadAttention)

    except ModuleNotFoundError:  # pragma: no cover
        pass
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to add flops_counter_hook: {e}")


_register_ptflops()
