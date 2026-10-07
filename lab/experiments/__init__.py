from . import alu, cache, mem


def registry():
    reg = {}
    reg.update(alu.registry())
    reg.update(mem.registry())
    reg.update(cache.registry())
    return reg
