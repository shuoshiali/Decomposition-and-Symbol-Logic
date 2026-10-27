import json
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter

random.seed(12)


@dataclass
class RawSample:
    """Raw data sample structure"""
    story_id: int
    example_id: int
    claim: str
    atomic_facts: Dict[str, str]
    symbolic_logic: str
    owner_label: Optional[str] = None
    id: Optional[int] = None


@dataclass
class SFTSample:
    """SFT training sample structure (compatible with LlamaFactory Alpaca format)"""
    instruction: str
    input: str
    output: str
    system: str = ""  # LlamaFactory supports the system field
    label: str = "standard"  # Label: standard | correction_chain


class LogicSFTDataProcessor:
    """
    Logical reasoning SFT data processor (LlamaFactory-compatible)
    Converts raw claim + atomic_facts + symbolic_logic into two-stage chained SFT data
    Output format: LlamaFactory Alpaca-format JSON array
    Adds a label field: standard (original correct data) | correction_chain (error-correction chain)
    """

    def __init__(self, raw_data_path: str, output_dir: str = "/mnt/agents/output/"):
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_samples: List[RawSample] = []
        self.sft_samples: List[SFTSample] = []

        # Placeholder templates (must exactly match inference-time format)
        self.STAGE_DECOMPOSE = "<stage>Decompose</stage>"
        self.STAGE_SYMBOLIZE = "<stage>Symbolize</stage>"
        self.STAGE_VERIFY = "<stage>Verify</stage>"
        self.TAG_ATOMIC_OPEN = "<atomic>"
        self.TAG_ATOMIC_CLOSE = "</atomic>"
        self.TAG_SYMBOL_OPEN = "<symbol>"
        self.TAG_SYMBOL_CLOSE = "</symbol>"
        self.TAG_ANSWER_OPEN = "<answer>"
        self.TAG_ANSWER_CLOSE = "</answer>"

        # LlamaFactory system prompt
        self.SYSTEM_PROMPT = (
            "You are a logical reasoning assistant. Follow the two-stage format strictly. "
            "First decompose the claim into minimal sufficient atomic facts, "
            "then generate the symbolic logic expression."
        )

    # ==================== 1. Data Loading ====================

    def load_data(self) -> None:
        """Load raw JSON data; supports .json or .jsonl"""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.raw_data_path}")

        raw_data = []
        if self.raw_data_path.suffix == ".jsonl":
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
        else:
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

        for item in raw_data:
            sample = RawSample(
                story_id=item.get("story_id", 0),
                example_id=item.get("example_id", 0),
                claim=item.get("claim", "").strip(),
                atomic_facts=item.get("atomic_facts", {}),
                symbolic_logic=item.get("symbolic_logic", "").strip(),
                owner_label=item.get("owner_label"),
                id=item.get("id")
            )
            # Basic filter: skip samples with empty claim or symbolic logic
            if sample.claim and sample.symbolic_logic:
                self.raw_samples.append(sample)

        print(f"[Load] Loaded {len(self.raw_samples)} valid raw samples from {self.raw_data_path}")

    # ==================== 2. Utility Methods ====================

    def _format_atomic_facts(self, atomic_facts: Dict[str, str]) -> str:
        """Format the atomic facts dict as ordered text (A1, A2, A3...)"""
        lines = []
        # Sort by A1, A2, A3
        for key in sorted(atomic_facts.keys()):
            lines.append(f"{key}: {atomic_facts[key]}")
        return "\n".join(lines)

    def _build_instruction(self, claim: str) -> str:
        """Build a LlamaFactory-compatible instruction (concise, no system prompt)"""
        return f"Claim: {claim}"

    def _build_input(self) -> str:
        """Build the input field (extensible for additional context)"""
        return (
            "Decompose the claim into minimal sufficient atomic facts, "
            "then generate the symbolic logic expression."
        )

    def _build_standard_output(self, atomic_facts: Dict[str, str], symbolic_logic: str) -> str:
        """Build the standard two-stage output format"""
        atomic_text = self._format_atomic_facts(atomic_facts)
        return (
            f"{self.STAGE_DECOMPOSE}\n"
            f"{self.TAG_ATOMIC_OPEN}\n"
            f"{atomic_text}\n"
            f"{self.TAG_ATOMIC_CLOSE}\n\n"
            f"{self.STAGE_SYMBOLIZE}\n"
            f"{self.TAG_SYMBOL_OPEN}{symbolic_logic}{self.TAG_SYMBOL_CLOSE}\n\n"
            f"{self.STAGE_VERIFY}\n"
            f"{self.TAG_ANSWER_OPEN}The claim is logically represented as above.{self.TAG_ANSWER_CLOSE}"
        )

    # ==================== 3. Standard Sample Construction (70%) ====================

    def build_standard_samples(self, ratio: float = 0.7) -> List[SFTSample]:
        """
        Build standard format-following samples
        Simulates: 70% come from DeepSeekV4 full-score answers (i.e., correct annotations in the raw data)
        label: "standard"
        """
        n = max(1, int(round(len(self.raw_samples) * ratio)))
        # Use all samples if the dataset is small
        selected = random.sample(self.raw_samples, min(n, len(self.raw_samples)))

        samples = []
        for raw in selected:
            sample = SFTSample(
                instruction=self._build_instruction(raw.claim),
                input=self._build_input(),
                output=self._build_standard_output(raw.atomic_facts, raw.symbolic_logic),
                system=self.SYSTEM_PROMPT,
                label="standard"  # Marked as original correct data
            )
            samples.append(sample)

        self.sft_samples.extend(samples)
        print(f"[Standard] Built {len(samples)} standard samples (ratio={ratio})")
        return samples

    # ==================== 4. Correction Chain Sample Construction (30%) ====================

    def _inject_redundancy(self, atomic_facts: Dict[str, str]) -> Dict[str, str]:
        """Inject a redundant atomic fact (e.g., add an already-implied fact)"""
        redundant = dict(atomic_facts)
        n = len(redundant)
        # Add a fact related to the claim but redundant
        redundant[f"A{n+1}"] = "This situation involves people and their family members."
        return redundant

    def _inject_missing_premise(self, atomic_facts: Dict[str, str]) -> Dict[str, str]:
        """Drop a key premise: randomly remove a non-conclusive premise (assume the last key is the conclusion)"""
        keys = list(atomic_facts.keys())
        if len(keys) <= 1:
            return atomic_facts

        # Drop the second-to-last key (assuming the last one is the conclusion A3)
        drop_key = keys[-2] if len(keys) >= 2 else keys[0]
        filtered = {k: v for k, v in atomic_facts.items() if k != drop_key}
        return filtered

    def _inject_wrong_connector(self, symbolic_logic: str) -> str:
        """Wrong logical connector: ∧→∨, →→↔"""
        wrong = symbolic_logic.replace("∧", "∨")
        wrong = wrong.replace("→", "↔")
        return wrong

    def _inject_negation_error(self, symbolic_logic: str, atomic_facts: Dict[str, str]) -> str:
        """Negation error: randomly negate one atomic proposition"""
        keys = list(atomic_facts.keys())
        if not keys:
            return symbolic_logic
        target = random.choice(keys)
        # Use regex exact match to avoid partial replacements (e.g., A1 vs. A10)
        pattern = re.compile(r'\b' + re.escape(target) + r'\b')
        return pattern.sub(f"¬{target}", symbolic_logic)

    def _get_correction_hint(self, error_type: str) -> str:
        """Return the correction hint for the given error type"""
        hints = {
            "redundancy": (
                "Correction: Remove redundant facts that are already implied by previous statements. "
                "Keep only minimal sufficient atomic facts."
            ),
            "missing_premise": (
                "Correction: Add missing necessary premises to make the reasoning complete. "
                "Ensure all required conditions for the conclusion are present."
            ),
            "wrong_connector": (
                "Correction: Fix logical connectors. Use ∧ (and) for joint conditions, "
                "→ (implies) for conditional relationships. Do not use ∨ or ↔ here."
            ),
            "negation_error": (
                "Correction: Remove incorrect negations. The symbolic logic must align "
                "with the atomic facts without introducing false negations."
            )
        }
        return hints.get(error_type, "Correction: Fix the logical error.")

    def build_correction_chain_samples(self, ratio: float = 0.3) -> List[SFTSample]:
        """
        Build correction-chain samples (30%)
        Teaches the model to: see an erroneous output, receive a correction hint, then produce the correct result
        This strengthens self-correction ability during inference
        label: "correction_chain"
        """
        n = max(1, int(round(len(self.raw_samples) * ratio)))
        selected = random.sample(self.raw_samples, min(n, len(self.raw_samples)))

        # Error injectors (Decompose-stage errors / Symbolize-stage errors)
        decompose_errors = [
            ("redundancy", self._inject_redundancy, "decompose"),
            ("missing_premise", self._inject_missing_premise, "decompose")
        ]
        symbolize_errors = [
            ("wrong_connector", self._inject_wrong_connector, "symbolize"),
            ("negation_error", self._inject_negation_error, "symbolize")
        ]

        samples = []
        for raw in selected:
            # Randomly choose the error stage
            if random.random() < 0.5 and len(raw.atomic_facts) > 1:
                error_name, error_fn, stage = random.choice(decompose_errors)
                wrong_atomic = error_fn(raw.atomic_facts)
                wrong_symbol = raw.symbolic_logic  # Symbolic logic left unchanged (based on erroneous atoms)

                # Build correction-chain output: wrong atoms -> correction hint -> correct atoms -> symbol stage
                wrong_output = (
                    f"{self.STAGE_DECOMPOSE}\n"
                    f"{self.TAG_ATOMIC_OPEN}\n"
                    f"{self._format_atomic_facts(wrong_atomic)}\n"
                    f"{self.TAG_ATOMIC_CLOSE}\n"
                    f"[{self._get_correction_hint(error_name)}]\n"
                    f"{self.TAG_ATOMIC_OPEN}\n"
                    f"{self._format_atomic_facts(raw.atomic_facts)}\n"
                    f"{self.TAG_ATOMIC_CLOSE}\n\n"
                    f"{self.STAGE_SYMBOLIZE}\n"
                    f"{self.TAG_SYMBOL_OPEN}{raw.symbolic_logic}{self.TAG_SYMBOL_CLOSE}\n\n"
                    f"{self.STAGE_VERIFY}\n"
                    f"{self.TAG_ANSWER_OPEN}The claim is logically represented as above.{self.TAG_ANSWER_CLOSE}"
                )
            else:
                error_name, error_fn, stage = random.choice(symbolize_errors)
                if error_name == "negation_error":
                    wrong_symbol = error_fn(raw.symbolic_logic, raw.atomic_facts)
                else:
                    wrong_symbol = error_fn(raw.symbolic_logic)

                # Build correction-chain output: correct atoms -> wrong symbol -> correction hint -> correct symbol
                wrong_output = (
                    f"{self.STAGE_DECOMPOSE}\n"
                    f"{self.TAG_ATOMIC_OPEN}\n"
                    f"{self._format_atomic_facts(raw.atomic_facts)}\n"
                    f"{self.TAG_ATOMIC_CLOSE}\n\n"
                    f"{self.STAGE_SYMBOLIZE}\n"
                    f"{self.TAG_SYMBOL_OPEN}{wrong_symbol}{self.TAG_SYMBOL_CLOSE}\n"
                    f"[{self._get_correction_hint(error_name)}]\n"
                    f"{self.TAG_SYMBOL_OPEN}{raw.symbolic_logic}{self.TAG_SYMBOL_CLOSE}\n\n"
                    f"{self.STAGE_VERIFY}\n"
                    f"{self.TAG_ANSWER_OPEN}The claim is logically represented as above.{self.TAG_ANSWER_CLOSE}"
                )

            sample = SFTSample(
                instruction=self._build_instruction(raw.claim),
                input=self._build_input(),
                output=wrong_output,
                system=self.SYSTEM_PROMPT,
                label="correction_chain"  # Marked as synthesized error-correction data
            )
            samples.append(sample)

        self.sft_samples.extend(samples)
        print(f"[Correction] Built {len(samples)} correction chain samples (ratio={ratio})")
        return samples

    # ==================== 5. Validation and Cleaning ====================

    def validate_samples(self) -> Tuple[List[SFTSample], List[Dict]]:
        """
        Validate format integrity of all samples:
        1. Contains all required placeholders
        2. Atomic facts and symbolic logic are non-empty
        3. Stage order is correct (Decompose -> Symbolize -> Verify)
        """
        valid = []
        invalid = []

        for idx, sample in enumerate(self.sft_samples):
            output = sample.output
            errors = []

            # Check required tags
            required_tags = [
                (self.STAGE_DECOMPOSE, "missing_decompose_stage"),
                (self.STAGE_SYMBOLIZE, "missing_symbolize_stage"),
                (self.STAGE_VERIFY, "missing_verify_stage"),
                (self.TAG_ATOMIC_OPEN, "missing_atomic_open"),
                (self.TAG_ATOMIC_CLOSE, "missing_atomic_close"),
                (self.TAG_SYMBOL_OPEN, "missing_symbol_open"),
                (self.TAG_SYMBOL_CLOSE, "missing_symbol_close"),
            ]
            for tag, err_name in required_tags:
                if tag not in output:
                    errors.append(err_name)

            # Check non-empty content (regex extraction)
            atomic_match = re.search(
                rf'{re.escape(self.TAG_ATOMIC_OPEN)}(.*?){re.escape(self.TAG_ATOMIC_CLOSE)}',
                output, re.DOTALL
            )
            symbol_match = re.search(
                rf'{re.escape(self.TAG_SYMBOL_OPEN)}(.*?){re.escape(self.TAG_SYMBOL_CLOSE)}',
                output, re.DOTALL
            )

            if atomic_match:
                atomic_content = atomic_match.group(1).strip()
                if not atomic_content or len(atomic_content) < 2:
                    errors.append("empty_atomic_content")
            else:
                errors.append("atomic_parse_failed")

            if symbol_match:
                symbol_content = symbol_match.group(1).strip()
                if not symbol_content or len(symbol_content) < 2:
                    errors.append("empty_symbol_content")
            else:
                errors.append("symbol_parse_failed")

            # Check stage order
            decompose_pos = output.find(self.STAGE_DECOMPOSE)
            symbolize_pos = output.find(self.STAGE_SYMBOLIZE)
            verify_pos = output.find(self.STAGE_VERIFY)

            if not (decompose_pos < symbolize_pos < verify_pos):
                errors.append("wrong_stage_order")

            if errors:
                invalid.append({
                    "index": idx,
                    "label": sample.label,
                    "errors": errors,
                    "output_preview": output[:200] + "..." if len(output) > 200 else output
                })
            else:
                valid.append(sample)

        self.sft_samples = valid
        print(f"[Validate] {len(valid)} valid, {len(invalid)} invalid removed")
        if invalid:
            print(f"[Validate] First invalid errors: {invalid[0]['errors']}")
        return valid, invalid

    def deduplicate(self) -> List[SFTSample]:
        """Deduplicate based on (instruction, output)"""
        seen = set()
        unique = []
        for sample in self.sft_samples:
            key = (sample.instruction, sample.output)
            if key not in seen:
                seen.add(key)
                unique.append(sample)

        removed = len(self.sft_samples) - len(unique)
        self.sft_samples = unique
        print(f"[Deduplicate] {len(unique)} unique (removed {removed} duplicates)")
        return unique

    def shuffle(self) -> List[SFTSample]:
        """Shuffle samples to prevent grouping of same-type samples"""
        random.shuffle(self.sft_samples)
        return self.sft_samples

    # ==================== 6. Saving and Statistics (LlamaFactory-Compatible) ====================

    def save(self, filename: str = "folio_sft_data.json") -> str:
        """
        Save as LlamaFactory-compatible Alpaca-format JSON array
        Includes label field: standard | correction_chain
        Also generates a dataset_info.json config file
        """
        output_path = self.output_dir / filename

        # Convert to LlamaFactory Alpaca-format JSON array (with label field)
        alpaca_data = []
        for sample in self.sft_samples:
            record = {
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output,
                "system": sample.system,
                "label": sample.label  # Added: distinguishes standard/correction-chain samples
            }
            alpaca_data.append(record)

        # Save as JSON array (required by LlamaFactory)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

        # Generate dataset_info.json (LlamaFactory dataset description)
        dataset_name = filename.replace(".json", "")
        dataset_info = {
            dataset_name: {
                "file_name": filename,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system"
                }
            }
        }

        info_path = self.output_dir / "dataset_info.json"
        # Merge if dataset_info.json already exists
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                existing_info = json.load(f)
            existing_info.update(dataset_info)
            dataset_info = existing_info

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        # Statistics (including label distribution)
        label_counts = Counter(s.label for s in self.sft_samples)

        stats = {
            "total_samples": len(self.sft_samples),
            "label_distribution": dict(label_counts),
            "avg_output_length": sum(len(s.output) for s in self.sft_samples) / max(1, len(self.sft_samples)),
            "avg_instruction_length": sum(len(s.instruction) for s in self.sft_samples) / max(1, len(self.sft_samples)),
            "dataset_info": dataset_info
        }

        stats_path = self.output_dir / filename.replace(".json", "_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # Also save a more readable demo file
        demo_path = self.output_dir / filename.replace(".json", "_demo.md")
        with open(demo_path, 'w', encoding='utf-8') as f:
            f.write("# LlamaFactory SFT Data Demo (First 3 Samples)\n\n")
            for i, s in enumerate(self.sft_samples[:3]):
                f.write(f"## Sample {i+1} (Label: {s.label})\n\n")
                f.write(f"**System:**\n{s.system}\n\n")
                f.write(f"**Instruction:**\n{s.instruction}\n\n")
                f.write(f"**Input:**\n{s.input}\n\n")
                f.write(f"**Output:**\n```\n{s.output}\n```\n\n")
                f.write("---\n\n")

        print(f"[Save] Saved {len(self.sft_samples)} samples to {output_path}")
        print(f"[Save] dataset_info.json saved to {info_path}")
        print(f"[Save] Stats saved to {stats_path}")
        print(f"[Save] Demo saved to {demo_path}")
        return str(output_path)

    # ==================== 7. Full Pipeline ====================

    def run(self,
            standard_ratio: float = 0.7,
            correction_ratio: float = 0.3,
            do_validate: bool = True,
            do_deduplicate: bool = True,
            do_shuffle: bool = True) -> str:
        """
        Full processing pipeline
        """
        print("=" * 60)
        print("Logic SFT Data Processing Pipeline (LlamaFactory Compatible)")
        print("=" * 60)

        self.load_data()
        self.build_standard_samples(ratio=standard_ratio)
        self.build_correction_chain_samples(ratio=correction_ratio)

        if do_validate:
            self.validate_samples()
        if do_deduplicate:
            self.deduplicate()
        if do_shuffle:
            self.shuffle()

        result_path = self.save()
        print("=" * 60)
        print("Pipeline completed.")
        print("=" * 60)
        return result_path


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Modify this path to your actual input file
    INPUT_FILE = "./FOLIO_ALL_Select.json"  # Raw data path
    OUTPUT_DIR = "./sft_data/"

    # Run the processor
    processor = LogicSFTDataProcessor(INPUT_FILE, OUTPUT_DIR)
    result = processor.run(
        standard_ratio=0.7,
        correction_ratio=0.3,
        do_validate=True,
        do_deduplicate=True,
        do_shuffle=True
    )

    print(f"\nFinal output file: {result}")
    print(f"\nLlamaFactory usage:")
    print(f"  1. Copy {result} and dataset_info.json to your LlamaFactory data directory")
    print(f"  2. In your training config, set dataset: logic_sft_data")