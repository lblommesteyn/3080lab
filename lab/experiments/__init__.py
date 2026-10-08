from . import alu, cache, mem, regfile


def registry():
    reg = {}
    reg.update(alu.registry())
    reg.update(mem.registry())
    reg.update(cache.registry())
    reg.update(regfile.registry())
    return reg
