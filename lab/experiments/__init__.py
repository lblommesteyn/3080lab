from . import alu, cache, mem, randk, realloc, regfile


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk, realloc):
        reg.update(mod.registry())
    return reg
