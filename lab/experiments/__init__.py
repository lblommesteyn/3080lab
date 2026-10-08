from . import alu, cache, gemv, gemv2, gemv3, gemv4, gemv5, l2map, mem, membw, randk, realloc, regfile, tensor


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc, gemv, gemv2, gemv3, gemv4, gemv5, membw, l2map, tensor):
        reg.update(mod.registry())
    return reg
