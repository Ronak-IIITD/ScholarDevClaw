"""Step 3 tests: mapper aliases cover nanoGPT + generic (Llama-style) patterns."""

from scholardevclaw.mapping.engine import MappingEngine, _fuzzy_match
from scholardevclaw.nano_specs import load_spec


def _engine_for(elements, spec_name):
    return MappingEngine(
        repo_analysis={"elements": elements, "imports": []},
        research_spec=load_spec(spec_name),
    )


def _el(name, line=1):
    return {"name": name, "type": "class", "file": "model.py", "line": line}


def test_fuzzy_alias_helpers():
    # Previously dead self.* keys now resolve via normalized lookup.
    assert _fuzzy_match("ln_1", "self.ln_1") is True
    assert _fuzzy_match("LlamaRMSNorm", "LayerNorm") is True
    assert _fuzzy_match("v_proj", "self.q_proj") is True
    assert _fuzzy_match("gate_proj", "class MLP") is True
    assert _fuzzy_match("Dropout", "LayerNorm") is False


def test_rmsnorm_hits_nanogpt_and_llama():
    engine = _engine_for([_el("LayerNorm", 18), _el("LlamaRMSNorm", 40)], "rmsnorm")
    result = engine.map()
    assert result.targets, "rmsnorm should hit LayerNorm / LlamaRMSNorm"
    assert result.confidence > 0


def test_qknorm_hits_llama_projections():
    elements = [_el("CausalSelfAttention", 29), _el("v_proj", 50), _el("o_proj", 55)]
    result = _engine_for(elements, "qknorm").map()
    assert len(result.targets) >= 2


def test_swiglu_hits_llama_mlp_sublayers():
    elements = [_el("MLP", 78), _el("gate_proj", 80)]
    result = _engine_for(elements, "swiglu").map()
    assert len(result.targets) >= 1


def test_rope_hits_wpe():
    # rope targets self.wpe / nn.Embedding; nn.Embedding hits via import tier,
    # wpe hits via fuzzy alias on element names.
    elements = [_el("wpe", 128), _el("GPT", 118)]
    result = _engine_for(elements, "rope").map()
    assert result.targets, "rope should hit wpe element"
