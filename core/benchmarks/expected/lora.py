EXPECTED_SYMBOLS = ("LoRALinear", "apply_lora")


def apply_lora(inputs, base, down, up):
    projected = [sum(value * weight for value, weight in zip(inputs, row)) for row in down]
    update = [sum(value * weight for value, weight in zip(projected, row)) for row in up]
    return [base_value + delta for base_value, delta in zip(base, update)]


class LoRALinear:
    def __call__(self, inputs, base, down, up):
        return apply_lora(inputs, base, down, up)
