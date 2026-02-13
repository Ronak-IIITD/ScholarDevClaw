from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ComponentMatch:
    name: str
    file: str
    line: int
    component_type: str


class PyTorchComponentDetector:
    def detect_modules(self, modules: List) -> List[ComponentMatch]:
        matches = []

        for module in modules:
            for cls in module.classes:
                if cls.is_nn_module:
                    matches.append(
                        ComponentMatch(
                            name=cls.name,
                            file=str(module.relative_path),
                            line=cls.line_number,
                            component_type="model",
                        )
                    )

        return matches

    def detect_layer_norms(self, modules: List) -> List[ComponentMatch]:
        matches = []

        for module in modules:
            source = module.path.read_text()

            for cls in module.classes:
                if "LayerNorm" in cls.name or "layer_norm" in cls.name:
                    matches.append(
                        ComponentMatch(
                            name=cls.name,
                            file=str(module.relative_path),
                            line=cls.line_number,
                            component_type="normalization",
                        )
                    )

        return matches

    def detect_attention(self, modules: List) -> List[ComponentMatch]:
        matches = []

        for module in modules:
            for cls in module.classes:
                if "Attention" in cls.name or "attention" in cls.name:
                    matches.append(
                        ComponentMatch(
                            name=cls.name,
                            file=str(module.relative_path),
                            line=cls.line_number,
                            component_type="attention",
                        )
                    )

        return matches
