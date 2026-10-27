import os
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from prompts import LOGICALITY_EVALUATION
from utils_GPT import APILLM


class IntegratedEvaluator:
    def __init__(self):
        self.model = APILLM()

        # Keep only the logicality metric
        self.metrics = {
            "logicality": "whether the symbolic logic correctly and accurately represents the logical relationship in the original claim based on the given atomic facts. Your answer should be ONLY \"true\" or \"false\".",
        }

    def set_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def set_output_file(self, path):
        self.output_file = path

    # ==================== LLM evaluation (Logicality) ====================
    def evaluate_llm(self, claim, atomic_facts=None, symbolic_logic=None):

        def atomic_str(data: dict) -> str:
            """Convert the atomic facts dictionary into the formatted string."""
            result = ""
            for key, val in data.items():
                result += f'{key}: "{val}",\n'
            result = "\n" + result.rstrip(",\n")
            return result

        # Logicality (claim-level; requires symbolic_logic)
        logicality = False
        if symbolic_logic and atomic_facts:
            user_prompt = LOGICALITY_EVALUATION.format(
                claim=claim,
                atomic_facts=atomic_str(atomic_facts),
                symbolic_logic=symbolic_logic
            )
            response = self.model.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fair evaluator. No additional commentary whatsoever."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                do_sample=True,
                temperature=0.01
            )
            content = response.strip().lower()
            if "true" in content and "false" not in content:
                logicality = True
            else:
                logicality = False

        return {
            "logicality": logicality
        }

    # ==================== Main evaluation loop (saves results incrementally) ====================
    def evaluate_all(self):
        data = self.data
        # Normalize to a list; supports a single object or a list of objects
        if isinstance(data, dict):
            data = [data]
            is_single = True
        else:
            is_single = False

        def reformat_atomic_facts(item):
            fact_list = item["subclaims"]
            new_fact_dict = {}
            # Generate A1, A2, ... key-value pairs
            for idx, fact in enumerate(fact_list, start=1):
                key = f"A{idx}"
                new_fact_dict[key] = fact
            # Replace the original list with the dictionary
            item["atomic_facts"] = new_fact_dict
            return item

        results = []

        for idx, item in enumerate(tqdm(data)):
            # if idx <= 447:
            #     continue
            claim = item['claim']
            claim_entities = item.get('claim_entities', None)

            # Obtain atomic_facts (compatible with the subclaims list format)
            if 'subclaims' in item and 'atomic_facts' not in item:
                item = reformat_atomic_facts(item)

            atomic_facts = item.get('atomic_facts', {})
            symbolic_logic = item.get('symbolic_logic', None)

            # ---------- Assign the lowest logicality value when atomic_facts or symbolic_logic is empty ----------
            if not atomic_facts or not symbolic_logic:
                item['logicality'] = False
            else:
                # ---------- LLM evaluation (Logicality) ----------
                llm_scores = self.evaluate_llm(
                    claim=claim,
                    atomic_facts=atomic_facts if atomic_facts else None,
                    symbolic_logic=symbolic_logic
                )
                item['logicality'] = llm_scores['logicality']
            item['atomicity'] = 0
            item['atomicity_fine_grained'] = []
            item['coverage'] = ""
            item['redundancy'] = ""
            item['fabrication'] = []
            if claim_entities is None:
                item['claim_entities'] = {"subjects": [], "objects": []}
            item['sub_claim_entities'] = [{"subjects": [], "objects": []}]

            # ---------- Save the current progress immediately ----------
            results.append(item)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                # Preserve the same structure as the input: a single object or a list of objects
                output_data = results[0] if is_single else results
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Evaluation completed. Results saved to: {self.output_file}")

def main():
    model_name = "Qwen3-8B"
    evaluator = IntegratedEvaluator()
    
    evaluator.set_data(f"./data/FactLens_{model_name}.json")
    evaluator.set_output_file(f"./results/FactLens_{model_name}_result.json")
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()