"""GRPO stage for LLaMA-Factory, delegated to TRL GRPOTrainer."""

import json
import random
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from datasets import Dataset

from ...extras import logging
from ...hparams import get_infer_args  # noqa: F401
from ...model import load_model, load_tokenizer
from ...data import get_template_and_fix_tokenizer
from .sacr_grpo_trainer import SACRGRPOTrainer, ComponentStore, make_sacr_reward_funcs
from .logic_verifier import LLMAssistedValidator, parse_data_item, load_json_robust
from .reward import DiReLogicReward

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments
    from transformers import Seq2SeqTrainingArguments

logger = logging.get_logger(__name__)

# ==================== Constants migrated from de_sft.py ====================
RANDOM_SEED = 12
random.seed(RANDOM_SEED)

FORMAT_TYPES = ["bullet", "compact", "json_like"]
MASK_NONE_RATIO = 0.20
MASK_LITE_RATIO = 0.30

SYSTEM_MESSAGE = (
    "You are a logical reasoning assistant with deep linguistic expertise. "
    "Your task is to perform logical atomic decomposition of natural language claims. "
    "You MUST follow the three-stage format strictly: "
    "(1) Parse: analyze syntactic structure to identify scope boundaries, "
    "logical connectives (XOR ⊕, NOR, AND ∧, OR ∨), conditional markers (If/then →), "
    "argument roles (ARG1, ARG2), and restrictor clauses; "
    "(2) Decompose: extract minimal sufficient atomic facts; "
    "(3) Symbolize: generate the formal symbolic logic expression. "
    "Use the provided syntactic clues as structural prior knowledge to resolve ambiguities."
)
# SYSTEM_MESSAGE = (
#     "You are a logical reasoning assistant. "
#     "Follow the two-stage format strictly. "
#     "First decompose the claim into minimal sufficient atomic facts, then generate the symbolic logic expression."
# )

SYSTEM_MESSAGE_TREE = (
    "You are a logical reasoning assistant with deep linguistic expertise. "
    "Your task is to perform logical atomic decomposition of natural language claims. "
    "You MUST follow the three-stage format strictly: "
    "(1) Parse: analyze syntactic structure to identify scope boundaries, "
    "logical connectives (XOR ⊕, NOR, AND ∧, OR ∨), conditional markers (If/then →), "
    "argument roles (ARG1, ARG2), and restrictor clauses; "
    "(2) Decompose: extract minimal sufficient atomic facts; "
    "(3) Symbolize: generate the formal symbolic logic expression. "
    "Use the provided syntactic clues as structural prior knowledge to resolve ambiguities.\n\n"
    "You MUST output strictly in the following XML format and nothing else:\n"
    "<stage>Parse</stage>\n"
    "<analysis>\n...your parse reasoning...\n</analysis>\n\n"
    "<stage>Decompose</stage>\n"
    "<atomic>\n...one atomic fact per line...\n</atomic>\n\n"
    "<stage>Symbolize</stage>\n"
    "<symbol>\n...symbolic logic expression...\n</symbol>\n\n"
    "Do NOT include any additional stages (e.g., Verify). "
    "Do NOT add any text, markdown, or explanations outside these XML tags."
)



# ==================== Feature extraction migrated from de_sft.py ====================

def extract_linguistic_features(dep_tree: str) -> Dict[str, Any]:
    """Extract structured linguistic clues from raw dependency-tree text
    (with semantic filtering and an atomicity guard added)."""
    features = {
        "logical_connectives": [],
        "conditionals": [],
        "arguments": [],
        "restrictions": [],
        "variables": [],
        "atomicity_guard": {"suppress": False, "reason": None}
    }

    lines = [l.strip() for l in dep_tree.split('\n') if l.strip() and '→' in l]

    # ==================== Tree-level index ====================
    token_info = {}  # idx -> {"word": str, "relations": set()}
    for line in lines:
        m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
        if m:
            rel, word, idx = m.group(1), m.group(2), m.group(3)
            token_info.setdefault(idx, {"word": word, "relations": set()})
            token_info[idx]["relations"].add(rel)

    sorted_items = sorted(token_info.items(), key=lambda x: int(x[0]))
    reconstructed = " ".join(info["word"] for _, info in sorted_items).lower()

    def _nearby_words(idx: str, window: int = 3) -> set:
        idx_int = int(idx)
        return {info["word"].lower() for other_idx, info in token_info.items()
                if abs(int(other_idx) - idx_int) <= window}

    # Whether the tree contains a verbal predicate (to distinguish NP structures from full sentences)
    verbal_rels = {"root", "cop", "xcomp", "ccomp", "advcl", "acl"}
    has_verb = any(
        any(r.startswith(vr) or vr in r for vr in verbal_rels for r in info["relations"])
        for info in token_info.values()
    )

    # ==================== Atomicity guard (global pattern detection) ====================
    math_patterns = [
        "sum of", "product of", "difference between", "ratio of",
        "total of", "total amount of", "average of", "mean of", "between",
        "percent change in", "percentage change in", "net change in",
        "adjusted balance", "beginning of period", "end of period",
        "case reserves", "total reserves", "cumulative",
    ]
    is_math_expr = any(pat in reconstructed for pat in math_patterns)

    exclusivity = {"only", "unique", "sole", "single"}
    has_exclusivity = any(w in reconstructed for w in exclusivity)
    has_relclause = any(
        any("acl" in r or "relcl" in r for r in info["relations"])
        for info in token_info.values()
    )

    if is_math_expr and not has_verb:
        features["atomicity_guard"] = {"suppress": True, "reason": "MATHEMATICAL_EXPRESSION"}
    elif has_exclusivity and has_relclause:
        features["atomicity_guard"] = {"suppress": True, "reason": "EXCLUSIVE_RESTRICTIVE_CLAUSE"}
    elif not has_verb and len(token_info) > 0:
        features["atomicity_guard"] = {"suppress": True, "reason": "NON_VERBAL_NP"}

    # ==================== Feature extraction with semantic filtering ====================
    for line in lines:
        line = line.strip()
        if not line or '→' not in line:
            continue

        # ---------- CONJ ----------
        if '→ CONJ:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ CONJ:\s*(.+)', line)
            if m:
                rel, token, idx, logic = m.group(1), m.group(2), m.group(3), m.group(4).strip()
                tok_lower = token.lower()
                nearby = _nearby_words(idx, window=2)

                # Filter 1: and/or inside mathematical expressions -> non-propositional coordination
                if tok_lower in {"and", "or"} and is_math_expr:
                    continue

                # Filter 2: coordination inside a prepositional phrase (sum of A and B / between A and B)
                if tok_lower in {"and", "or"} and (nearby & {"of", "with", "between", "among"}):
                    if not has_verb:
                        continue

                # Filter 3: and/or inside an exclusivity qualifier + relative clause
                if tok_lower in {"and", "or"} and has_exclusivity and has_relclause:
                    continue

                # Filter 4: pure nominal coordination (all conj items are nominal, no verbal coordination)
                if rel in {"cc", "conj"} and tok_lower in {"and", "or"}:
                    conj_items = [
                        info for oidx, info in token_info.items()
                        if any("conj" in r for r in info["relations"]) and oidx != idx
                    ]
                    if conj_items:
                        nominal_rels = {"nsubj", "dobj", "iobj", "nmod", "compound", "det", "amod"}
                        all_nominal = all(
                            any(nr in r for r in info["relations"] for nr in nominal_rels)
                            or len(info["relations"]) == 0
                            for info in conj_items
                        )
                        any_verbal_conj = any(
                            any(vr in r for r in info["relations"] for vr in verbal_rels)
                            for info in conj_items
                        )
                        if all_nominal and not any_verbal_conj:
                            continue

                features["logical_connectives"].append({
                    "relation": rel, "token": token, "idx": idx,
                    "type": "CONJ", "logic": logic
                })

        # ---------- XOR ----------
        elif '→ XOR:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ XOR:\s*(.+)', line)
            if m:
                if features["atomicity_guard"]["suppress"]:
                    continue
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "XOR", "logic": m.group(4).strip()
                })

        # ---------- DISJ ----------
        elif '→ DISJ:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ DISJ:\s*(.+)', line)
            if m:
                if features["atomicity_guard"]["suppress"]:
                    continue
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "DISJ", "logic": m.group(4).strip()
                })

        # ---------- NOR ----------
        elif '→ NOR' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                if features["atomicity_guard"]["suppress"]:
                    continue
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "NOR", "logic": "¬∨"
                })

        # ---------- COND / ARG / RESTRICT / VAR (original logic unchanged) ----------
        elif '→ COND:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ COND:\s*(.+)', line)
            if m:
                features["conditionals"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "logic": m.group(4).strip()
                })
        elif '→ ARG1' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["arguments"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "role": "ARG1"
                })
        elif '→ ARG2' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["arguments"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "role": "ARG2"
                })
        elif '→ RESTRICT' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["restrictions"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3)
                })
        elif '→ VAR:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ VAR:\s*(.+)', line)
            if m:
                features["variables"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "var": m.group(4).strip()
                })

    return features


def _format_bullet(features: Dict, lite: bool = False) -> str:
    """Format A: structured list (same as de_sft.py)."""
    sections = []
    
    if features["logical_connectives"]:
        lines = ["Logical Connectives (scope markers):"]
        for c in features["logical_connectives"]:
            lines.append(f"  • {c['token']}[{c['idx']}] ({c['relation']}): {c['type']} → {c['logic']}")
        sections.append("\n".join(lines))
    
    if features["conditionals"]:
        lines = ["Conditional Markers (P→Q boundaries):"]
        for c in features["conditionals"]:
            lines.append(f"  • {c['token']}[{c['idx']}] ({c['relation']}): {c['logic']}")
        sections.append("\n".join(lines))
    
    if not lite:
        if features["arguments"]:
            lines = ["Core Arguments (entity tracking):"]
            for a in features["arguments"]:
                lines.append(f"  • {a['token']}[{a['idx']}] ({a['relation']}): {a['role']}")
            sections.append("\n".join(lines))
        
        if features["restrictions"]:
            lines = ["Restrictor Clauses (variable binding):"]
            for r in features["restrictions"]:
                lines.append(f"  • {r['token']}[{r['idx']}] ({r['relation']}): domain restrictor")
            sections.append("\n".join(lines))
        
        if features["variables"]:
            lines = ["Bound Variables (anaphora):"]
            for v in features["variables"]:
                lines.append(f"  • {v['token']}[{v['idx']}] ({v['relation']}): bound to {v['var']}")
            sections.append("\n".join(lines))
    
    return "\n\n".join(sections) if sections else "(No linguistic features)"


def _format_compact(features: Dict, lite: bool = False) -> str:
    """Format B: compact lines (same as de_sft.py)."""
    clues = []
    
    for c in features["logical_connectives"]:
        clues.append(f"{c['token']}[{c['idx']}]:{c['type']}({c['logic']})")
    for c in features["conditionals"]:
        clues.append(f"{c['token']}[{c['idx']}]:COND({c['logic']})")
    
    if not lite:
        for a in features["arguments"]:
            clues.append(f"{a['token']}[{a['idx']}]:{a['role']}")
        for r in features["restrictions"]:
            clues.append(f"{r['token']}[{r['idx']}]:RESTRICT")
        for v in features["variables"]:
            clues.append(f"{v['token']}[{v['idx']}]:VAR({v['var']})")
    
    return "CLUES: " + "; ".join(clues) if clues else "CLUES: none"


def _format_json_like(features: Dict, lite: bool = False) -> str:
    """Format C: pseudo-JSON (same as de_sft.py)."""
    obj = {}
    
    if features["logical_connectives"]:
        obj["conn"] = [
            {"t": c["token"], "i": int(c["idx"]), "r": c["relation"], "type": c["type"], "logic": c["logic"]}
            for c in features["logical_connectives"]
        ]
    if features["conditionals"]:
        obj["cond"] = [
            {"t": c["token"], "i": int(c["idx"]), "r": c["relation"], "logic": c["logic"]}
            for c in features["conditionals"]
        ]
    
    if not lite:
        if features["arguments"]:
            obj["args"] = [
                {"t": a["token"], "i": int(a["idx"]), "r": a["relation"], "role": a["role"]}
                for a in features["arguments"]
            ]
        if features["restrictions"]:
            obj["restr"] = [
                {"t": r["token"], "i": int(r["idx"]), "r": r["relation"]}
                for r in features["restrictions"]
            ]
        if features["variables"]:
            obj["vars"] = [
                {"t": v["token"], "i": int(v["idx"]), "r": v["relation"], "bind": v["var"]}
                for v in features["variables"]
            ]
    
    return json.dumps(obj, ensure_ascii=False) if obj else "{}"


def format_linguistic_features(features: Dict, fmt: str, lite: bool = False) -> str:
    """Unified entry: render linguistic clues in the requested format (same as de_sft.py)."""
    if fmt == "bullet":
        return _format_bullet(features, lite)
    elif fmt == "compact":
        return _format_compact(features, lite)
    elif fmt == "json_like":
        return _format_json_like(features, lite)
    else:
        return _format_bullet(features, lite)

def build_forward_system(features: Dict, mask_type: str, fmt: str) -> str:
    """
    Build a system prompt identical to SFT.
    mask_type: none | lite | full
    """
    guard = features.get("atomicity_guard", {})

    # Atomicity guard override: strong directive for claims that must not be decomposed
    if guard.get("suppress"):
        reason = guard.get("reason", "")
        guard_msg = (
            f"\n\n[ATOMICITY GUARD: {reason}] "
            f"This claim expresses a SINGLE atomic proposition. "
            f"Do NOT decompose it into multiple atomic facts under any circumstances. "
            f"Output exactly ONE atomic fact and ONE symbolic expression."
        )
        base = SYSTEM_MESSAGE_TREE + guard_msg

        if mask_type == "none":
            return base

        clues = format_linguistic_features(features, fmt, lite=(mask_type == "lite"))
        return (
            f"{base}\n\n"
            f"[Syntactic Clues for This Claim]\n{clues}\n\n"
            f"Use these clues only for scope validation, but respect the ATOMICITY GUARD above."
        )

    # Original logic unchanged
    if mask_type == "none":
        return SYSTEM_MESSAGE_TREE

    clues = format_linguistic_features(features, fmt, lite=(mask_type == "lite"))

    return (
        f"{SYSTEM_MESSAGE_TREE}\n\n"
        f"[Syntactic Clues for This Claim]\n{clues}\n\n"
        f"Use these clues to determine operator scope and argument binding accurately."
    )


def build_forward_input(claim: str) -> str:
    """Build a user input identical to SFT (tree moved into system; input keeps only the claim)."""
    return (
        f"Claim: {claim}"
    )


def _render_prompt(instruction: str) -> str:
    """Plain-text rendering (used by build_grpo_prompts.py stage2 pre-rendering, same as qwen3_nothink)."""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def _build_prompt_dataset(finetuning_args, tokenizer, data_args):
    """Render prompts with the tokenizer's chat template to match SFT character-for-character."""
    items = []
    with open(finetuning_args.grpo_prompt_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Get the template (same as SFT stage, to align special tokens)
    _ = get_template_and_fix_tokenizer(tokenizer, data_args)

    for it in raw:
        claim = it["claim"]
        tree = it.get("tree", "") or it.get("dependency_tree", "")
        
        # Reuse a pre-rendered prompt from the JSON (e.g., curriculum mode) if present
        if it.get("prompt") and getattr(finetuning_args, "grpo_use_prebuilt_prompt", True):
            prompt = it["prompt"]
        else:
            # ========== Re-render the prompt ==========
            if tree:
                features = extract_linguistic_features(tree)
                
                # Reproduce the SFT mask distribution
                r = random.random()
                if r < MASK_NONE_RATIO:
                    mask_type = "none"
                elif r < MASK_NONE_RATIO + MASK_LITE_RATIO:
                    mask_type = "lite"
                else:
                    mask_type = "full"
                
                fmt = random.choice(FORMAT_TYPES)
                system_prompt = build_forward_system(features, mask_type, fmt)
                
                # Same Alpaca-style concat as SFT: instruction + "\n" + input
                instruction = (
                    "Decompose the claim into minimal sufficient atomic facts, "
                    "then generate the symbolic logic expression."
                )
                input_text = build_forward_input(claim)
                user_content = instruction + "\n" + input_text
                
            else:
                system_prompt = SYSTEM_MESSAGE
                user_content = (
                    f"Claim: {claim}\n\n"
                    "Extract minimal sufficient atomic facts and generate the symbolic logic expression.\n"
                    "Output ONLY these two XML tags, no explanations, no analysis, no other stages:\n"
                    "<atomic>\nA1: [first atomic fact]\nA2: [second atomic fact]\n...</atomic>\n"
                    "<symbol>[symbolic logic using A1, A2, ... and connectives ∧∨¬→↔]</symbol>"
                )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Length check (applied to both pre-rendered and re-rendered prompts)
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len > finetuning_args.grpo_max_prompt_length:
            logger.warning_rank0(
                f"[DiReRL] Prompt too long ({prompt_len} > {finetuning_args.grpo_max_prompt_length}), "
                f"tree will be truncated! Claim: {claim[:50]}..."
            )
        
        items.append({"prompt": prompt, "claim": claim, "tree": tree})

    # Diagnostics
    if items:
        has_tree = any(it.get("tree", "") for it in items)
        logger.info_rank0(
            f"[DiReRL] First prompt preview (has_tree={has_tree}, len={len(items[0]['prompt'])}):\n"
            f"{items[0]['prompt'][:400]}\n"
            f"[DiReRL] Sample with tree: {len(items[0].get('tree', '')) > 0}\n"
            f"[DiReRL] Tree preview: {items[0].get('tree', '')[:200]}"
        )

    return Dataset.from_list(items)


def _build_teacher_index(finetuning_args: "FinetuningArguments") -> dict:
    teachers = {}
    for raw in load_json_robust(finetuning_args.grpo_teacher_file):
        teacher, _ = parse_data_item(raw)
        if teacher and teacher.symbolic_logic:
            # Same normalization as the training data (simple conjunction -> pure ∧)
            teachers[teacher.claim] = teacher
    logger.info_rank0(f"[DiReRL] Teacher index: {len(teachers)} claims (normalized).")
    return teachers


def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: Optional[Any] = None,
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    from trl import GRPOConfig, GRPOTrainer
    import numpy as _np
    import torch as _th
    import os as _os

    # ==================== Tier1: env overrides + seed control ====================
    def _ov(env_name, cast, current):
        v = _os.environ.get(env_name)
        return cast(v) if v is not None else current

    run_tag = _os.environ.get("DIRE_RUN_TAG", "")
    seed = int(_os.environ.get("DIRE_SEED", "12"))
    random.seed(seed)
    _np.random.seed(seed)
    _th.manual_seed(seed)
    _th.cuda.manual_seed_all(seed)

    lr = _ov("DIRE_LR", float, training_args.learning_rate)
    beta = _ov("DIRE_BETA", float, finetuning_args.grpo_beta)
    n_gen = _ov("DIRE_K", int, finetuning_args.grpo_num_generations)
    temperature = _ov("DIRE_TEMPERATURE", float, finetuning_args.grpo_temperature)
    scale_rewards = _os.environ.get(
        "DIRE_SCALE_REWARDS", getattr(finetuning_args, "grpo_scale_rewards", "group"))
    loss_type = _os.environ.get(
        "DIRE_LOSS_TYPE", getattr(finetuning_args, "grpo_loss_type", "dr_grpo"))
    w_atomic = _ov("DIRE_W_ATOMIC", float, finetuning_args.dire_reward_w_atomic)
    w_symbolic = _ov("DIRE_W_SYMBOLIC", float, finetuning_args.dire_reward_w_symbolic)
    dense_fb = _ov("DIRE_DENSE_FALLBACK", float, finetuning_args.dire_reward_dense_fallback)
    output_dir = (training_args.output_dir + "/" + run_tag) if run_tag else training_args.output_dir
    logger.info_rank0(
        f"[DiReRL/Tier1] tag={run_tag or 'default'} seed={seed} lr={lr} beta={beta} "
        f"K={n_gen} T={temperature} scale_rewards={scale_rewards} loss={loss_type} "
        f"w_a={w_atomic} w_s={w_symbolic} dense_fb={dense_fb} out={output_dir}"
    )

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)

    # Fix TRL + PEFT warnings_issued compatibility issue
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Build the dataset (rendered with the tokenizer to match the SFT distribution)
    dataset = _build_prompt_dataset(finetuning_args, tokenizer, data_args)
    teacher_index = _build_teacher_index(finetuning_args)

    reward_fn = DiReLogicReward(
        teacher_index=teacher_index,
        llm=None,
        w_atomic=w_atomic,                      # Tier1: env-overridable
        w_symbolic=w_symbolic,                  # Tier1: env-overridable
        use_llm_fallback=finetuning_args.dire_reward_use_llm_fallback,
        dense_fallback=dense_fb,                # Tier1: env-overridable
        group_size=n_gen,                       # Tier1 fix: zero_std stats follow K
    )

    # ========== SACR mode (research use, optional) ==========
    store = ComponentStore(["gate", "atomic", "symbolic"])
    reward_funcs = make_sacr_reward_funcs(reward_fn, store)   # 3 component functions

    grpo_config = GRPOConfig(
        output_dir=output_dir,              # Tier1: run_tag isolation
        learning_rate=lr,                   # Tier1: env-overridable
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        bf16=training_args.bf16,
        report_to=training_args.report_to,
        seed=seed,                          # Tier1: env-overridable
        # ===== GRPO-specific =====
        num_generations=n_gen,              # Tier1: env-overridable
        beta=beta,                          # Tier1: env-overridable
        temperature=temperature,            # Tier1: env-overridable
        max_prompt_length=finetuning_args.grpo_max_prompt_length,
        max_completion_length=finetuning_args.grpo_max_completion_length,
        loss_type=loss_type,                # Tier1: env-overridable
        scale_rewards=scale_rewards,        # Tier1: env-overridable + getattr fallback
        use_vllm=finetuning_args.grpo_use_vllm,
        vllm_mode=finetuning_args.grpo_vllm_mode,
        vllm_gpu_memory_utilization=finetuning_args.grpo_vllm_gpu_mem,
        log_completions=False,
    )

    trainer = SACRGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,          # <- 3 functions, not [reward_fn]
        args=grpo_config,
        train_dataset=dataset,
        sacr_store=store,                   # <- pass the store
        callbacks=callbacks or [],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info_rank0(f"[DiReRL] Reward stats (last): {reward_fn.last_stats}")