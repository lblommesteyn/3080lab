from . import alu


def registry():
    reg = {}
    reg.update(alu.registry())
    return reg
