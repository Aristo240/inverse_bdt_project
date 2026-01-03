import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Outcome:
    features: np.ndarray 
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features, dtype=float)
        else:
            self.features = self.features.astype(float)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON saving."""
        return {
            "features": self.features.tolist(),
            "description": self.description
        }

@dataclass
class Lottery:
    outcomes: List[Outcome]
    probs: np.ndarray

    def __post_init__(self):
        if not isinstance(self.probs, np.ndarray):
            self.probs = np.array(self.probs, dtype=float)
        
        if len(self.probs) != len(self.outcomes):
            raise ValueError(f"Probs length ({len(self.probs)}) != Outcomes length ({len(self.outcomes)})")

        total = self.probs.sum()
        if total <= 0:
            raise ValueError("Probabilities must sum to > 0")
        if not np.isclose(total, 1.0):
            self.probs = self.probs / total

    def get_stats(self):
        X = np.stack([o.features for o in self.outcomes])
        p = self.probs.reshape(-1, 1)

        mean_x = np.sum(X * p, axis=0)
        mean_sq_x = np.sum((X**2) * p, axis=0)
        var_x = mean_sq_x - mean_x**2
        
        var_x = np.maximum(var_x, 0.0)
        return mean_x, var_x

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON saving."""
        return {
            "outcomes": [o.to_dict() for o in self.outcomes],
            "probs": self.probs.tolist()
        }

    def to_prompt_string(self, label: str) -> str:
        text = f"Option {label}:\n"
        for p, out in zip(self.probs, self.outcomes):
            if p > 0:
                pct = int(round(p * 100))
                util, harm, rep = out.features
                text += (f"  - {pct}% chance: Utility {util:.1f}/10, "
                         f"Safety Violation {harm:.1f}/10, Reputation Risk {rep:.1f}/10.\n")
        return text