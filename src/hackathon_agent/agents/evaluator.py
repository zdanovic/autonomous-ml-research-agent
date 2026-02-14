"""
Evaluator Agent.

Responsibility: rigorous metric calculation and leaderboard management.
"""
from typing import Any

from ..core.agent import Agent, AgentState
from ..schema.experiment import ExperimentResult, LeaderboardEntry


class EvaluatorState(AgentState):
    leaderboard: list[LeaderboardEntry] = []

class Evaluator(Agent[EvaluatorState]):
    """
    Agent that validates results and maintains the leaderboard.
    """
    
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate new results and update leaderboard.
        """
        # Get latest results from context (usually populated by MLEngineer)
        # In a real async system, this might consume a queue. 
        # Here we look at the 'history' or a 'new_results' key.
        
        # Taking "history" from context, but ideally we'd filter for "just finished"
        history: list[ExperimentResult] = context.get("history", [])
        
        # Sync leaderboard
        new_entries = 0
        current_ids = {e.experiment_id for e in self.state.leaderboard}
        
        for res in history:
            if res.experiment_id not in current_ids and res.success:
                entry = LeaderboardEntry(
                    experiment_id=res.experiment_id,
                    cv_score=res.cv_score,
                    timestamp=res.timestamp
                )
                self.state.leaderboard.append(entry)
                new_entries += 1
                
        # Sort leaderboard
        self.state.leaderboard.sort(key=lambda x: x.cv_score)
        
        if new_entries > 0:
            best = self.state.leaderboard[0]
            self.log(f"Updated Leaderboard. Top Score: {best.cv_score:.4f} (Exp: {best.experiment_id})")
            
        return {
            "leaderboard": self.state.leaderboard,
            "best_score": self.state.leaderboard[0].cv_score if self.state.leaderboard else float("inf")
        }
