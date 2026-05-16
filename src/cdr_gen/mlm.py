"""Thin ESM-2 t12 masked-LM wrapper.

Exposes per-position AA probabilities for the guided sampler. Follows the
same model / dim conventions as ProtePilot's `src/esm2_features.py` (t12,
960-dim VH+VL concat) but does NOT depend on ProtePilot — the sampler
needs its own lightweight interface to avoid cross-repo coupling.

Semantics:
  - `joint_sequence(vh, vl, cdr_span)` returns (joint_str, cdr_joint_pos)
    where cdr_joint_pos is the offset of the CDR's first residue inside
    the joint VH + <linker> + VL string.
  - `position_probs(joint_str, position)` returns a (20,) numpy array
    of renormalized probabilities over STANDARD_AA at that position,
    with a mask token placed at that position for inference.
"""

from __future__ import annotations

import numpy as np

from cdr_gen.schema import STANDARD_AA

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
DEFAULT_LINKER = "GGGGSGGGGSGGGGS"


class MaskedLM:
    """Load-once wrapper with opt-in mocking for tests."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        linker: str = DEFAULT_LINKER,
        _model=None,
        _tokenizer=None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.linker = linker
        self._model = _model
        self._tokenizer = _tokenizer
        self._mask_id: int | None = None
        self._aa_ids: dict[str, int] | None = None
        if _tokenizer is not None:
            self._build_aa_index()

    # --- lazy load ---------------------------------------------------------

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("transformers + torch are required") from e
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name).eval()
        try:
            model = model.to(self.device)
        except Exception:
            pass
        self._tokenizer = tok
        self._model = model
        self._build_aa_index()

    def _build_aa_index(self) -> None:
        vocab = self._tokenizer.get_vocab()
        missing = [a for a in STANDARD_AA if a not in vocab]
        if missing:
            raise RuntimeError(f"ESM-2 vocab missing AAs: {missing}")
        self._aa_ids = {aa: vocab[aa] for aa in STANDARD_AA}
        mask = getattr(self._tokenizer, "mask_token", None)
        if mask is None or mask not in vocab:
            raise RuntimeError("tokenizer has no mask token")
        self._mask_id = vocab[mask]

    # --- core API ---------------------------------------------------------

    def joint_sequence(self, vh: str, vl: str) -> tuple[str, int]:
        """Return (joint_str, vl_offset). VH starts at index 0, VL at vl_offset."""
        joint = (vh or "") + self.linker + (vl or "")
        vl_offset = len(vh) + len(self.linker)
        return joint, vl_offset

    def position_probs(
        self, joint_seq: str, position: int, temperature: float = 1.0,
    ) -> np.ndarray:
        """Return (20,) probs over STANDARD_AA at position after masking it.

        Special tokens (BOS/EOS) are handled automatically; position is
        in raw-sequence coordinates (0-indexed within joint_seq).
        """
        self._lazy_load()
        import torch

        tok = self._tokenizer.encode(joint_seq, add_special_tokens=True, return_tensors="pt")
        try:
            tok = tok.to(self.device)
        except Exception:
            pass
        tok_pos = position + 1  # account for BOS
        masked = tok.clone()
        masked[0, tok_pos] = self._mask_id

        with torch.no_grad():
            logits = self._model(masked).logits  # (1, L, V)
        scaled = logits[0, tok_pos] / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1).detach().cpu().numpy()

        probs_20 = np.array([probs[self._aa_ids[aa]] for aa in STANDARD_AA])
        s = probs_20.sum()
        if s <= 0:
            return np.ones(20, dtype=np.float32) / 20
        return (probs_20 / s).astype(np.float32)

    def score_sequence_log_lik(self, joint_seq: str, positions: list[int]) -> float:
        """Mean log-prob of the residues at `positions` under ESM-2 masked LM.

        Used to compute per-design ESM perplexity as a quality proxy.
        """
        if not positions:
            return 0.0
        log_probs = []
        for p in positions:
            probs = self.position_probs(joint_seq, p)
            aa = joint_seq[p]
            if aa in STANDARD_AA:
                idx = STANDARD_AA.index(aa)
                log_probs.append(float(np.log(max(probs[idx], 1e-12))))
        if not log_probs:
            return 0.0
        return sum(log_probs) / len(log_probs)
