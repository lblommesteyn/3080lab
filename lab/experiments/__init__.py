from . import alu, cache, gemv, mem, randk, realloc, regfile


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc, gemv):
        reg.update(mod.registry())
    return reg
