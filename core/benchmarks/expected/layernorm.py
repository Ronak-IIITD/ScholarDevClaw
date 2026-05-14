import math

EXPECTED_SYMBOLS = ("LayerNorm", "layer_norm")


def layer_norm(values, eps=1e-5):
    mean = sum(values) / max(len(values), 1)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
    denom = math.sqrt(variance + eps)
    return [(value - mean) / denom for value in values]


class LayerNorm:
    def __call__(self, values):
        return layer_norm(values)
