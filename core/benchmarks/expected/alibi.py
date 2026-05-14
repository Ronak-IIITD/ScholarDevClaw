EXPECTED_SYMBOLS = ("ALiBi", "build_alibi_bias")


def build_alibi_bias(n_heads, length):
    return [[-(head + 1) * position for position in range(length)] for head in range(n_heads)]


class ALiBi:
    def __call__(self, n_heads, length):
        return build_alibi_bias(n_heads, length)
