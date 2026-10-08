from . import alu, cache, gemv, gemv2, mem, randk, realloc, regfile


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc, gemv, gemv2):
        reg.update(mod.registry())
    return reg
