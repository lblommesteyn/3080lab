from . import alu, cache, frontend, pairs, gemv, gemv2, gemv3, gemv4, gemv5, l2map, mem, membw, randk, realloc, regfile, smem, tensor


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc, gemv, gemv2, gemv3, gemv4, gemv5, membw, l2map, tensor, frontend, smem,
                pairs):
        reg.update(mod.registry())
    return reg
