import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
from trl import GRPOTrainer

# ==================== Tier1: unified config entry (defaults identical to the original) ====================
try:
    import dire_config as _C
except ImportError:
    class _C:
        ATOMIC_BOOST, CLIP_EPSILON = 1.2, 0.2
        METRICS_PATH, RUN_TAG = "", "default"

ATOMIC_ADV_BOOST = _C.ATOMIC_BOOST        # Advantage boost factor for the atomic stage (default 1.2)
CLIP_EPSILON_DEFAULT = _C.CLIP_EPSILON    # Final PPO-clip fallback (default 0.2)


# ==================== Component reward staging (import from reward.py, or redundantly defined here) ====================

class ComponentStore:
    """Per-component reward staging shared by the three reward functions."""

    def __init__(self, names: List[str]):
        self.names = list(names)
        self.reset()

    def reset(self):
        self.values: Dict[str, List[float]] = {n: [] for n in self.names}

    def append(self, name: str, v: float):
        self.values[name].append(float(v))

    def to_tensor(self, device) -> torch.Tensor:
        n = len(self.values[self.names[0]])
        cols = [self.values[name] for name in self.names]
        data = [[cols[c][i] for c in range(len(self.names))] for i in range(n)]
        return torch.tensor(data, dtype=torch.float32, device=device)


def make_sacr_reward_funcs(reward_fn, store: ComponentStore) -> List[Callable]:
    """Split DiReLogicReward into three additive component rewards."""
    # Tier1 fix: cache key changed from comp to (claim, comp) to avoid cross-claim
    # collisions on identical text
    cache: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

    def components(prompts, completions, **kwargs):
        claims = kwargs.get("claim", [""] * len(completions))
        out = []
        for claim, comp in zip(claims, completions):
            key = (claim or "", comp)
            if key not in cache:
                if len(cache) > 50000:
                    cache.clear()
                cache[key] = reward_fn.decompose(claim or "", comp)
            out.append(cache[key])
        return out

    def _make(idx: int, name: str):
        def fn(prompts, completions, **kwargs):
            vals = [c[idx] for c in components(prompts, completions, **kwargs)]
            for v in vals:
                store.append(name, v)
            return vals
        fn.__name__ = f"sacr_{name}"
        return fn

    return [_make(0, "gate"), _make(1, "atomic"), _make(2, "symbolic")]


# ==================== Segment mapping (XML-adapted) ====================

def _split_segments(text: str) -> List[Tuple[str, int, int]]:
    """Segment an XML-format completion: key in {A1..An, SYMBOLIC, STRUCT}."""
    segs, pos = [], 0
    in_atomic = False
    
    for ln in text.splitlines(keepends=True):
        stripped = ln.strip()
        
        # <atomic> opening tag
        if re.match(r'<atomic[^>]*>', stripped, re.IGNORECASE):
            in_atomic = True
            segs.append(("STRUCT", pos, pos + len(ln)))
            pos += len(ln)
            continue
        
        # </atomic> closing tag
        if re.match(r'</atomic>', stripped, re.IGNORECASE):
            in_atomic = False
            segs.append(("STRUCT", pos, pos + len(ln)))
            pos += len(ln)
            continue
        
        # <symbol> block (single- or multi-line)
        symbol_match = re.match(r'(<symbol[^>]*>)(.*?)(</symbol>)', stripped, re.IGNORECASE)
        if symbol_match:
            tag_open = symbol_match.group(1)
            content = symbol_match.group(2)
            tag_close = symbol_match.group(3)
            
            segs.append(("STRUCT", pos, pos + len(tag_open)))
            pos += len(tag_open)
            if content:
                segs.append(("SYMBOLIC", pos, pos + len(content)))
                pos += len(content)
            segs.append(("STRUCT", pos, pos + len(tag_close)))
            pos += len(ln) - len(tag_open) - len(content) - len(tag_close)
            continue
        
        # Inside an <atomic> block: match lines like "A1: xxx"
        if in_atomic:
            m = re.match(r'^(A\d+)\s*[:=.\s]\s*(.+)$', stripped)
            if m:
                key = m.group(1)
                key_pos = ln.find(key)
                start = pos + key_pos if key_pos >= 0 else pos
                segs.append((key, start, pos + len(ln)))
            else:
                segs.append(("STRUCT", pos, pos + len(ln)))
        else:
            # Outside blocks: check for a bare symbolic expression
            if any(c in stripped for c in '∧∨¬→↔') and re.search(r'\bA\d+\b', stripped):
                segs.append(("SYMBOLIC", pos, pos + len(ln)))
            else:
                segs.append(("STRUCT", pos, pos + len(ln)))
        
        pos += len(ln)
    
    return segs


def _token_spans(tokenizer, ids: List[int], text: str) -> List[Tuple[int, int, str]]:
    """Map character segments to token spans (t_start, t_end, key). Boundary error ~ +/-1 token."""
    n = len(ids)
    bounds = []
    for key, c0, _c1 in _split_segments(text):
        t0 = min(len(tokenizer.encode(text[:c0], add_special_tokens=False)), n)
        bounds.append((t0, key))
    bounds.sort()
    spans = []
    for j, (t0, key) in enumerate(bounds):
        t1 = bounds[j + 1][0] if j + 1 < len(bounds) else n
        if t1 > t0:
            spans.append((t0, t1, key))
    return spans


# ==================== SACR Trainer ====================

class SACRGRPOTrainer(GRPOTrainer):
    """Two hooks: component-decoupled advantage computation + segment routing;
    the advantage is no longer unsqueezed and broadcast in the loss."""

    def __init__(self, *args, sacr_store: Optional[ComponentStore] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sacr_store = sacr_store

    def _generate_and_score_completions(self, inputs):
        if self._sacr_store is not None:
            self._sacr_store.reset()
        out = super()._generate_and_score_completions(inputs)
        if self._sacr_store is None or not self._sacr_store.values.get("gate"):
            return out
        
        comp = self._sacr_store.to_tensor(out["completion_ids"].device)
        out["advantages"] = self._build_token_advantages(out, comp)

        # ---- Tier1: SACR mechanism metrics (segment routing stats + advantage dispersion) ----
        if _C.METRICS_PATH:
            import json as _json
            _rt = getattr(self, "_sacr_routed", {"sym": 0, "atom": 0, "total": 0})
            self._sacr_routed = {"sym": 0, "atom": 0, "total": 0}
            _adv = out["advantages"]
            with open(_C.METRICS_PATH, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps({
                    "src": "sacr", "run": _C.RUN_TAG,
                    "gate_mean": float(comp[:, 0].mean()),
                    "atomic_mean": float(comp[:, 1].mean()),
                    "symbolic_mean": float(comp[:, 2].mean()),
                    "adv_std": float(_adv.std()),
                    "routed_sym_frac": _rt["sym"] / max(1, _rt["total"]),
                    "routed_atom_frac": _rt["atom"] / max(1, _rt["total"]),
                }) + "\n")
        return out

    def _build_token_advantages(self, out: Dict[str, torch.Tensor], comp: torch.Tensor) -> torch.Tensor:
        B, C = comp.shape
        G = self.num_generations
        g = comp.view(-1, G, C)
        mean = g.mean(dim=1, keepdim=True)
        # Tier1 fix: the original `bool("global")` was also True, so the scale_rewards
        # config was effectively ignored; now explicitly mapped: group/True -> in-group
        # std normalization; none/False -> no normalization
        _sr = getattr(self, "scale_rewards", True)
        use_std = _sr if isinstance(_sr, bool) else (str(_sr).lower() not in ("none", "false"))
        std = g.std(dim=1, keepdim=True) if use_std else torch.ones_like(mean)
        adv = ((g - mean) / (std + 1e-4)).view(B, C)

        completion_ids, mask = out["completion_ids"], out["completion_mask"]
        L = completion_ids.shape[1]
        tok_adv = torch.zeros(B, L, dtype=torch.float32, device=comp.device)
        tok = self.processing_class
        
        for i in range(B):
            idx = mask[i].nonzero(as_tuple=True)[0]
            ids = completion_ids[i, idx].tolist()
            text = tok.decode(ids, skip_special_tokens=False)
            spans = _token_spans(tok, ids, text)
            line = torch.zeros(len(ids), dtype=torch.float32, device=comp.device)
            _rt = getattr(self, "_sacr_routed", None)
            if _rt is None:
                _rt = self._sacr_routed = {"sym": 0, "atom": 0, "total": 0}
            for s, e, key in spans:
                if key == "SYMBOLIC":
                    line[s:e] = adv[i, 2]
                    _rt["sym"] += (e - s)
                elif key != "STRUCT":
                    line[s:e] = adv[i, 1] * ATOMIC_ADV_BOOST  # Tier1: overridable via DIRE_ATOMIC_BOOST
                    _rt["atom"] += (e - s)
            _rt["total"] += len(ids)
            tok_adv[i, idx] = adv[i, 0] + line
            
            # Optional: print segment mapping for the first 2 samples only (safe version)
            if i < 2 and False:  # set to True to enable debug output
                print(f"\n[SACR DEBUG] Sample {i}")
                print(f"  Gate={adv[i,0]:.3f} Atomic={adv[i,1]:.3f} Symbolic={adv[i,2]:.3f}")
                for s, e, key in spans:
                    seg_val = (adv[i, 0] + (adv[i, 2] if key == "SYMBOLIC" else 
                              (adv[i, 1] if key != "STRUCT" else 0)))
                    print(f"  [{key:8s}] tokens {s:3d}:{e:3d} -> {seg_val:.3f}")
        
        return tok_adv

    def _compute_loss(self, model, inputs):
        """Compute log_probs directly from the model output instead of relying on
        _get_per_token_logps. Compatible with TRL 0.18-0.24 and future versions."""
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
    
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    
        # Model logits
        logits = model(input_ids, attention_mask=attention_mask).logits
        # Keep only the logits for the completion (from the last prompt token to the second-to-last completion token)
        logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
        # log_probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather the log_prob at each completion token position
        per_token_logps = torch.gather(log_probs, dim=2, index=completion_ids.unsqueeze(-1)).squeeze(-1)
    
        # KL term
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits
                    ref_logits = ref_logits[:, prompt_ids.shape[1] - 1 : -1, :]
                    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    ref_per_token_logps = torch.gather(ref_log_probs, dim=2, index=completion_ids.unsqueeze(-1)).squeeze(-1)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_logits = model(input_ids, attention_mask=attention_mask).logits
                        ref_logits = ref_logits[:, prompt_ids.shape[1] - 1 : -1, :]
                        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                        ref_per_token_logps = torch.gather(ref_log_probs, dim=2, index=completion_ids.unsqueeze(-1)).squeeze(-1)
            
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        else:
            per_token_kl = None
        
        # SACR: advantages are already [B, L]
        advantages = inputs["advantages"]
        
        # old_per_token_logps
        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        
        # PPO-clip
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        epsilon = getattr(self, "epsilon", None)
        if epsilon is None:
            epsilon = getattr(self, "clip_epsilon", None)
        if epsilon is None:
            # Fall back to config
            epsilon = getattr(getattr(self, "args", None), "clip_epsilon", CLIP_EPSILON_DEFAULT)
            if epsilon == CLIP_EPSILON_DEFAULT:  # still the default; try config
                epsilon = getattr(getattr(self, "config", None), "clip_epsilon", CLIP_EPSILON_DEFAULT)
        
        coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Aggregation
        loss_type = getattr(self, "loss_type", "grpo")
        if loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * completion_mask.shape[1])
        else:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        
        return loss