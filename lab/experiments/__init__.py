from . import alu, mem


def registry():
    reg = {}
    reg.update(alu.registry())
    reg.update(mem.registry())
    return reg
