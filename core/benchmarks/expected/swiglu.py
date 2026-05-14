EXPECTED_SYMBOLS = ("SwiGLU", "swiglu")


def swiglu(x_gate, x_value):
    return [(1 / (1 + 2.718281828 ** (-gate))) * value for gate, value in zip(x_gate, x_value)]


class SwiGLU:
    def __call__(self, x_gate, x_value):
        return swiglu(x_gate, x_value)
