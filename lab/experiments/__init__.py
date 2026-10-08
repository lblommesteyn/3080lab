from . import alu, cache, gemv, gemv2, gemv3, gemv4, mem, randk, realloc, regfile


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc, gemv, gemv2, gemv3, gemv4):
        reg.update(mod.registry())
    return reg
