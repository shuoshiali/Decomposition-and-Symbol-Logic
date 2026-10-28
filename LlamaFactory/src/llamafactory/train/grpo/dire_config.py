import os


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# ==================== Experiment record (shared by workflow / reward / trainer) ====================
SEED = _i("DIRE_SEED", 42)
RUN_TAG = os.environ.get("DIRE_RUN_TAG", "")
METRICS_PATH = os.environ.get("DIRE_METRICS_PATH", "")

# ==================== SACR (used by sacr_grpo_trainer.py) ====================
ATOMIC_BOOST = _f("DIRE_ATOMIC_BOOST", 1.2)   # Advantage boost factor for the atomic stage (previously hardcoded 1.2)
CLIP_EPSILON = _f("DIRE_CLIP_EPSILON", 0.2)   # Final PPO-clip fallback (previously hardcoded 0.2)

# ==================== Reward (used by reward.py) ====================
W_ATOMIC = _f("DIRE_W_ATOMIC", 0.4)           # Matches YAML dire_reward_w_atomic
W_SYMBOLIC = _f("DIRE_W_SYMBOLIC", 0.6)       # Matches YAML dire_reward_w_symbolic
DENSE_FALLBACK = _f("DIRE_DENSE_FALLBACK", 0.3)
GATE_PENALTY = _f("DIRE_GATE_PENALTY", -0.2)
VALIDATOR_ALPHA = _f("DIRE_VALIDATOR_ALPHA", 0.90)  # LLMAssistedValidator alpha
GROUP_SIZE = _i("DIRE_GROUP_SIZE", 8)  # Fallback only; the workflow passes grpo_num_generations

# ==================== Tier2: internal reward component weights (used by reward.py) ====================
# atomic component = d_score * adequacy * count_match * text_coverage (product chain stages)
ADEQ_SINGLE = _f("DIRE_ADEQ_SINGLE", 0.01)     # adequacy: score when a claim needs multiple atoms but only 1 is given
UNDER_DIFF1 = _f("DIRE_UNDER_DIFF1", 0.3)      # under-decomposition (1 atom missing)
UNDER_DIFF2 = _f("DIRE_UNDER_DIFF2", 0.05)     # under-decomposition (2 atoms missing)
OVER_DIFF1 = _f("DIRE_OVER_DIFF1", 0.85)       # over-decomposition (1 extra atom)
OVER_DIFF2 = _f("DIRE_OVER_DIFF2", 0.6)        # over-decomposition (2 extra atoms)
COV_JACCARD = _f("DIRE_COV_JACCARD", 0.6)      # text_coverage: word jaccard weight
COV_ENTITY = _f("DIRE_COV_ENTITY", 0.4)        # text_coverage: entity coverage weight
ENTITY_MATCH_TH = _f("DIRE_ENTITY_MATCH_TH", 0.6)  # Entity-match similarity threshold
FIDELITY_GATE = _f("DIRE_FIDELITY_GATE", 0.05)     # Fidelity hard gate (below this, whole item gets GATE_PENALTY)
DENSE_FID_MIN = _f("DIRE_DENSE_FID_MIN", 0.5)      # Minimum fidelity required to trigger dense_fallback

# ==================== GRPO (read directly by the env override layer in workflow.py; documentation only) ====================
# DIRE_LR              Learning rate (defaults to YAML learning_rate = 1e-6)
# DIRE_BETA            KL coefficient (defaults to YAML grpo_beta = 0.02)
# DIRE_K               Group size (defaults to YAML grpo_num_generations = 8)
# DIRE_TEMPERATURE     Sampling temperature (defaults to YAML grpo_temperature = 1.0)
# DIRE_SCALE_REWARDS   group | none (defaults to YAML grpo_scale_rewards, fallback group)
# DIRE_LOSS_TYPE       dr_grpo | grpo | bnpo (defaults to YAML grpo_loss_type)