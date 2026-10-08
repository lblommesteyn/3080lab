from . import alu, cache, mem, randk, regfile


def registry():
    reg = {}
    for mod in (alu, mem, cache, regfile, randk):
        reg.update(mod.registry())
    return reg
