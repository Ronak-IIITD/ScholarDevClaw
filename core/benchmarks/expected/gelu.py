import math

EXPECTED_SYMBOLS = ("gelu",)


def gelu(value):
    return 0.5 * value * (1.0 + math.tanh(0.7978845608 * (value + 0.044715 * value**3)))
