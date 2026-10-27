import json
from typing import List, Dict, Any, Optional


class ScoreCalculator:
    """Claim verification score aggregator."""
    
    LEVEL_MAP = {"high": 3, "medium": 2, "low": 1}
    BOOL_MAP = {True: 1, False: 0, "true": 1, "false": 0}
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.raw_data = data
        self.scores = []
        self._calculate_all()
    
    def _map_level(self, value: Optional[str]) -> int:
        """Map high/medium/low to 3/2/1; empty value -> 0."""
        if not value:
            return 0
        return self.LEVEL_MAP.get(str(value).lower().strip(), 0)
    
    def _map_bool(self, value: Any) -> int:
        """Map true/false to 1/0; empty value -> 0."""
        if value is None:
            return 0
        key = value if isinstance(value, (bool, str)) else str(value).lower()
        return self.BOOL_MAP.get(key, 0)
    
    def _calc_fabrication(self, fab_list: Optional[List]) -> float:
        """Map the fabrication list and average; empty list/None -> 0."""
        if not fab_list:
            return 0.0
        scores = [self._map_level(item) for item in fab_list]
        return sum(scores) / len(scores)
    
    def _calculate_single(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Compute the score for a single sample."""
        return {
            "id": item.get("id"),
            "atomicity": float(item.get("atomicity", 0) or 0),
            "coverage": self._map_level(item.get("coverage")),
            "fabrication": self._calc_fabrication(item.get("fabrication")),
            "redundancy": self._map_level(item.get("redundancy")),
            "logicality": self._map_bool(item.get("logicality"))
        }
    
    def _calculate_all(self):
        self.scores = [self._calculate_single(item) for item in self.raw_data]
    
    def get_all_scores(self) -> List[Dict[str, float]]:
        """Return detailed scores for all samples."""
        return self.scores
    
    def get_averages(self) -> Dict[str, float]:
        """Compute the average score for each metric."""
        if not self.scores:
            return {}
        keys = ["atomicity", "coverage", "fabrication", "redundancy", "logicality"]
        return {k: round(sum(s[k] for s in self.scores) / len(self.scores), 6) for k in keys}
    
    def print_report(self):
        print("Average scores:")
        for k, v in self.get_averages().items():
            print(f"  • {k}: {v:.6f}")
        print("=" * 55)


if __name__ == '__main__':
    model_name = "Qwen3-8B"
    path = f"./results/FactLens_{model_name}_result.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    calc = ScoreCalculator(data)
    calc.print_report()