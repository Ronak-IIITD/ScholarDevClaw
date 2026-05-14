EXPECTED_SYMBOLS = ("FlashAttention", "flash_attention")


def flash_attention(scores, values):
    return [sum(weight * value for weight, value in zip(row, values)) for row in scores]


class FlashAttention:
    def __call__(self, scores, values):
        return flash_attention(scores, values)
