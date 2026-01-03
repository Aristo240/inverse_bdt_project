import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    def __init__(self, base_dir="experiments/logs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Generate unique Run ID based on time
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.base_dir, f"run_{self.run_id}.json")
        
        print(f"Experiment Logging initialized: {self.log_file}")

    def save_results(self, 
                     config: Dict[str, Any], 
                     lotteries_A: list, 
                     lotteries_B: list, 
                     choices: np.ndarray, 
                     results: Dict[str, Any]):
        """
        Saves the entire state of the experiment to JSON.
        """
        
        # Structure the data
        data = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "config": config
            },
            "results": results,
            "data": []
        }
        
        # Save individual trials
        for i in range(len(choices)):
            trial = {
                "trial_idx": i,
                "lottery_A": lotteries_A[i].to_dict(),
                "lottery_B": lotteries_B[i].to_dict(),
                "choice": int(choices[i]), # 1 for A, 0 for B
                "choice_label": "Option A" if choices[i] == 1 else "Option B"
            }
            data["data"].append(trial)
            
        # Write to disk
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Results saved to {self.log_file}")