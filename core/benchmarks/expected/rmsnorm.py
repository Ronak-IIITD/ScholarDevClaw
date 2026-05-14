import math

EXPECTED_SYMBOLS = ("RMSNorm", "rms_norm")


def rms_norm(values, eps=1e-8):
    scale = math.sqrt(sum(value * value for value in values) / max(len(values), 1) + eps)
    return [value / scale for value in values]


class RMSNorm:
    def __call__(self, values):
        return rms_norm(values)
