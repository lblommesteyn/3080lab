from . import alu, cache, gemv, gemv2, gemv3, gemv4, gemv5, mem, randk, realloc, regfile


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc, gemv, gemv2, gemv3, gemv4, gemv5):
        reg.update(mod.registry())
    return reg
