import os
import json
import ast
import jaro
import itertools
from bert_score import BERTScorer
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

from prompts import ENTITY_EXTRACTOR_PROMPT
from prompts import COLLECTIVE_SUB_CLAIM_EVALUATION, INDIVIDUAL_SUB_CLAIM_EVALUATION, LOGICALITY_EVALUATION
from utils_GPT import APILLM


class IntegratedEvaluator:
    def __init__(self):
        self.model = APILLM()
        self.scorer = BERTScorer(
            model_type="./google-bert/bert-base-uncased",
            num_layers=12,
            lang="en"
        )

        # Metric definitions (from sub_claim_evaluator)
        self.metrics = {
            "redundancy": "if the sub-claims contain redundant or repeated information among them, i.e. multiple semantically equivalent sub-claims. Your answer should indicate whether the sub-claims have \"low\", \"medium\" or \"high\" redundancy.",
            "fabrication": "if the sub-claim shows a degree of fabrication with respect to the original claim i.e. how much new information is added which was not present in the original claim. Note this is not to be judged according to the factuality of the original claims or subclaims. Your answer should indicate whether the sub-claim has \"low\", \"medium\" or \"high\" fabrication.",
            "logicality": "whether the symbolic logic correctly and accurately represents the logical relationship in the original claim based on the given atomic facts. Your answer should be ONLY \"true\" or \"false\".",
        }

    def set_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def set_output_file(self, path):
        self.output_file = path

    # ==================== Entity extraction (sub_claims only; claim entities are no longer extracted) ====================
    def get_sub_claim_entities(self, sub_claims):
        '''
        Get entities (subjects, objects) for sub-claims only.
        Claim entities are expected to be pre-extracted in the dataset.
        '''
        def get_entities_helper(c):
            user_prompt = ENTITY_EXTRACTOR_PROMPT.format(claim=c)
            response = self.model.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "Your answer should be strictly be in form of a dictionary with keys: subjects and objects"
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                do_sample=True,
                temperature=0.01
            )
            content = json.loads(response)
            return content

        entities = {}
        for sub_claim in sub_claims:
            entities[sub_claim] = get_entities_helper(sub_claim)
        return entities

    def entity_match(self, x, y):
        a = x.lower().strip()
        b = y.lower().strip()
        return jaro.jaro_winkler_metric(a, b)

    def calculate_atomicity(self, s_i, o_i):
        if len(s_i) == 1 and len(o_i) == 1:
            return "atomic"
        elif len(s_i) == 1 and len(o_i) > 1:
            return "non-atomic-1"
        else:
            return "non-atomic-2"

    def calculate_coverage(self, S, O, sub_claims, entities):
        e = []
        for sub_claim in sub_claims:
            s_i = entities[sub_claim]["subjects"]
            e.extend(s_i)
            o_i = entities[sub_claim]["objects"]
            e.extend(o_i)

        e = list(set(e))
        E = S + O
        non_coverage = 0
        for e_i in E:
            is_covered = False
            for e_j in e:
                if self.entity_match(e_i, e_j) > 0.6:
                    is_covered = True
                    break
            if not is_covered:
                non_coverage += 1

        if non_coverage <= 0.25 * (len(S) + len(O)):
            return "high"
        elif non_coverage <= 0.75 * (len(S) + len(O)):
            return "medium"
        else:
            return "low"

    def compound_atomicity(self, atomicity):
        if len(atomicity) == 0:
            return 1
        score = 0
        for a in atomicity:
            score += 3 if a == "atomic" else 2 if a == "non-atomic-1" else 1
        return score / len(atomicity)

    # ==================== Automated evaluation (Atomicity + Coverage) ====================
    def evaluate_automated(self, claim, atomic_facts_dict, claim_entities):
        sub_claims = list(atomic_facts_dict.values())
        
        # Use the pre-extracted claim_entities from the dataset directly; no LLM call needed
        claim_entity = {
            "subjects": claim_entities["subjects"],
            "objects": claim_entities["objects"]
        }
        
        # Extract entities for sub_claims only
        entities = self.get_sub_claim_entities(sub_claims)
        # Add claim_entities to the dictionary for coverage computation
        entities[claim] = claim_entities

        S = claim_entities["subjects"]
        O = claim_entities["objects"]

        atomicity_fine_grained = []
        sub_claim_entity = []

        for sub_claim in sub_claims:
            s_i = entities[sub_claim]["subjects"]
            o_i = entities[sub_claim]["objects"]
            sub_claim_entity.append({
                "subjects": s_i,
                "objects": o_i
            })
            atomicity_fine_grained.append(self.calculate_atomicity(s_i, o_i))

        coverage = self.calculate_coverage(S, O, sub_claims, entities)

        return {
            "atomicity": self.compound_atomicity(atomicity_fine_grained),
            "atomicity_fine_grained": atomicity_fine_grained,
            "coverage": coverage
        }, claim_entity, sub_claim_entity

    # ==================== LLM evaluation (Fabrication + Redundancy + Logicality) ====================
    def evaluate_llm(self, claim, sub_claims, atomic_facts=None, symbolic_logic=None):

        def atomic_str(data: dict) -> str:
            """Convert the atomic facts dictionary into the formatted string."""
            result = ""
            for key, val in data.items():
                result += f'{key}: "{val}",\n'
            result = "\n" + result.rstrip(",\n")
            return result
        def evaluate_metric(prompt_template, metric_desc, claim_text, sub_claim_text):
            user_prompt = prompt_template.format(
                metrics=metric_desc,
                claim=claim_text,
                sub_claims=sub_claim_text
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
            return response.strip().lower()

        # Redundancy (claim-level)
        redundancy = ""
        if sub_claims:
            resp = evaluate_metric(
                COLLECTIVE_SUB_CLAIM_EVALUATION,
                self.metrics['redundancy'],
                claim,
                sub_claims
            )
            if "high" in resp.lower():
                redundancy = "high"
            elif "medium" in resp.lower():
                redundancy = "medium"
            else:
                redundancy = "low"

        # Fabrication (sub-claim-level, evaluated one by one)
        fabrication = []
        if sub_claims:
            for sub_claim in sub_claims:
                fab_resp = evaluate_metric(
                    INDIVIDUAL_SUB_CLAIM_EVALUATION,
                    self.metrics['fabrication'],
                    claim,
                    sub_claim
                )
                # Parse into low/medium/high
                if "high" in fab_resp.lower():
                    fabrication.append("high")
                elif "medium" in fab_resp.lower():
                    fabrication.append("medium")
                else:
                    fabrication.append("low")

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
            "redundancy": redundancy,
            "fabrication": fabrication,
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
            # if idx<=447:
            #     continue
            claim = item['claim']

            # Obtain sub_claims (compatible with the atomic_facts dictionary format)
            if 'subclaims' in item and 'atomic_facts' not in item:
                item = reformat_atomic_facts(item)

            if 'atomic_facts' in item:
                sub_claims = list(item['atomic_facts'].values())
            else:
                sub_claims = []

            atomic_facts = item.get('atomic_facts', {})
            symbolic_logic = item.get('symbolic_logic', None)
            # Read the pre-extracted claim_entities directly from the dataset
            claim_entities = item.get('claim_entities', None)

            # ---------- Assign the lowest value to all metrics when atomic_facts is empty ----------
            if not atomic_facts:
                item['atomicity'] = 0
                item['atomicity_fine_grained'] = []
                item['coverage'] = ""
                item['redundancy'] = ""
                item['fabrication'] = []
                item['logicality'] = False
                # Keep claim_entities if present in the dataset; otherwise set to empty
                if claim_entities is None:
                    item['claim_entities'] = {"subjects": [], "objects": []}
                item['sub_claim_entities'] = [{"subjects": [], "objects": []}]
            
            else:
                # Check whether the dataset contains pre-extracted claim_entities
                if claim_entities is None:
                    raise ValueError(
                        f"Item {idx} missing 'claim_entities'. "
                        f"Please pre-extract claim entities and add them to the dataset."
                    )
                
                # ---------- 1. Automated evaluation (Atomicity + Coverage) ----------
                automated_score, claim_entity, sub_claim_entity = self.evaluate_automated(
                    claim=claim,
                    atomic_facts_dict=atomic_facts,
                    claim_entities=claim_entities
                )
                item['atomicity'] = automated_score['atomicity']
                item['atomicity_fine_grained'] = automated_score['atomicity_fine_grained']
                item['coverage'] = automated_score['coverage']
                item['claim_entities'] = claim_entity
                item['sub_claim_entities'] = sub_claim_entity

                # ---------- 2. LLM evaluation (Fabrication + Redundancy + Logicality) ----------
                llm_scores = self.evaluate_llm(
                    claim=claim,
                    sub_claims=sub_claims,
                    atomic_facts=atomic_facts if atomic_facts else None,
                    symbolic_logic=symbolic_logic
                )
                item['redundancy'] = llm_scores['redundancy']
                item['fabrication'] = llm_scores['fabrication']
                item['logicality'] = llm_scores['logicality']

            # ---------- 3. Save the current progress immediately ----------
            results.append(item)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                # Preserve the same structure as the input: a single object or a list of objects
                output_data = results[0] if is_single else results
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Evaluation completed. Results saved to: {self.output_file}")

def main():
    model_name = "Qwen3-8B"
    evaluator = IntegratedEvaluator()
    
    # The input JSON must already contain the claim_entities field
    evaluator.set_data(f"./data/FactLens_{model_name}.json")
    evaluator.set_output_file(f"./results/FactLens_{model_name}_result.json")
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()