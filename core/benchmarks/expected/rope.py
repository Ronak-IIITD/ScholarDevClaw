import math

EXPECTED_SYMBOLS = ("RoPE", "apply_rope")


def apply_rope(pair, position, theta=10000.0):
    angle = position / theta
    return [
        pair[0] * math.cos(angle) - pair[1] * math.sin(angle),
        pair[0] * math.sin(angle) + pair[1] * math.cos(angle),
    ]


class RoPE:
    def __call__(self, pair, position):
        return apply_rope(pair, position)
