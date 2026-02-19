"""
Hackathon Optimizer Agent - Main orchestration module.

This agent autonomously:
1. Analyzes current state (diagnose overfitting, country imbalance)
2. Selects optimal strategies based on signals
3. Executes strategies and tracks results
4. Iterates until target is reached or max iterations
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .analyzer import Analyzer, DiagnosticReport
from .executor import Executor, ExperimentResult
from .knowledge_base import KnowledgeBase
from .strategist import Strategist


class HackathonOptimizerAgent:
    """
    Self-improving ML optimization agent for hackathons.
    
    Usage:
        agent = HackathonOptimizerAgent(
            train_df=train,
            test_df=test,
            target_score=0.18,
        )
        agent.run(max_iterations=10)
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "construction_cost_per_m2_usd",
        target_score: float = 0.18,
        submissions_dir: str = "submissions",
        experiments_path: str | None = None,
        preds_dir: str | None = None,
    ):
        self.train = train_df
        self.test = test_df
        self.target_col = target_col
        self.target_score = target_score
        
        # Initialize components
        self.kb = KnowledgeBase(
            experiments_path=experiments_path,
            preds_dir=preds_dir,
        )
        
        self.analyzer = Analyzer(experiments_df=self.kb.get_experiments())
        self.strategist = Strategist()
        self.executor = Executor(
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            submissions_dir=submissions_dir,
            knowledge_base=self.kb,
        )
        
        self.iteration = 0
        self.best_score = float("inf")
        self.best_result: ExperimentResult | None = None
        self.history: list[dict] = []
        
    def run(
        self,
        max_iterations: int = 10,
        strategies_per_iteration: int = 2,
        verbose: bool = True,
    ) -> ExperimentResult | None:
        """
        Run optimization loop.
        
        Args:
            max_iterations: Maximum number of strategy iterations
            strategies_per_iteration: How many strategies to try per iteration
            verbose: Print progress
            
        Returns:
            Best ExperimentResult or None
        """
        self._print_banner()
        
        for self.iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  🔄 ITERATION {self.iteration}/{max_iterations}")
                print(f"{'='*60}")
            
            # Step 1: Analyze current state
            report = self._diagnose()
            
            if verbose:
                self.analyzer.print_report(report)
            
            # Step 2: Select strategies
            strategies = self.strategist.select_next_strategy(
                report, 
                max_strategies=strategies_per_iteration
            )
            
            if not strategies:
                if verbose:
                    print("\n  ℹ️ No more applicable strategies. Stopping.")
                break
            
            if verbose:
                self.strategist.print_strategy_plan(strategies, report)
            
            # Step 3: Execute strategies
            for strategy in strategies:
                result = self.executor.run_strategy(strategy)
                self.strategist.mark_executed(strategy.name)
                
                # Track best
                if result.cv_score < self.best_score:
                    self.best_score = result.cv_score
                    self.best_result = result
                    
                    if verbose:
                        print(f"\n  🎉 NEW BEST: {result.cv_score:.5f}")
                
                # Log to history
                self.history.append({
                    "iteration": self.iteration,
                    "strategy": strategy.name,
                    "cv_score": result.cv_score,
                    "submission": result.submission_path,
                    "timestamp": datetime.now().isoformat(),
                })
                
                # Check if target reached
                if result.cv_score <= self.target_score:
                    if verbose:
                        print(f"\n  🏆 TARGET REACHED! CV={result.cv_score:.5f}")
                    self._print_summary()
                    return result
        
        # Print final summary
        self._print_summary()
        return self.best_result
    
    def _diagnose(self) -> DiagnosticReport:
        """Generate diagnostic report for current state."""
        # Get current best CV score
        current_cv = self.best_score if self.best_score < float("inf") else 0.25
        
        # Get OOF predictions if available
        oof = None
        y_true = None
        countries = None
        
        if self.best_result and self.best_result.oof_predictions is not None:
            oof = self.best_result.oof_predictions
            y_true = self.train[self.target_col].values
            countries = self.train["country"].values
        
        return self.analyzer.generate_diagnostic_report(
            cv_score=current_cv,
            public_score=None,  # We don't know public score during optimization
            oof_predictions=oof,
            y_true=y_true,
            countries=countries,
        )
    
    def run_single_strategy(self, strategy_name: str) -> ExperimentResult | None:
        """Run a specific strategy by name."""
        strategy = self.strategist.get_strategy_by_name(strategy_name)
        if strategy is None:
            print(f"Strategy '{strategy_name}' not found.")
            return None
        
        return self.executor.run_strategy(strategy)
    
    def get_available_strategies(self) -> list[str]:
        """Get list of available strategy names."""
        return list(self.strategist.strategies.keys())
    
    def _print_banner(self) -> None:
        """Print agent startup banner."""
        print("\n" + "=" * 60)
        print("  🚀 HACKATHON OPTIMIZER AGENT")
        print("=" * 60)
        print(f"  Target Score: {self.target_score}")
        print(f"  Train Size:   {len(self.train)}")
        print(f"  Test Size:    {len(self.test)}")
        print(f"  Countries:    {self.train['country'].unique().tolist()}")
        print("=" * 60)
    
    def _print_summary(self) -> None:
        """Print final optimization summary."""
        print("\n" + "=" * 60)
        print("  📊 OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        print(f"\n  Iterations: {self.iteration}")
        print(f"  Strategies Tried: {len(self.strategist.execution_history)}")
        
        if self.best_result:
            print("\n  🏆 Best Result:")
            print(f"     Strategy: {self.best_result.strategy_name}")
            print(f"     CV Score: {self.best_result.cv_score:.5f}")
            print(f"     Submission: {self.best_result.submission_path}")
            
            target_gap = self.best_result.cv_score - self.target_score
            if target_gap <= 0:
                print("\n  ✅ TARGET ACHIEVED!")
            else:
                print(f"\n  📈 Gap to target: {target_gap:.5f}")
        
        # Results table
        self.executor.print_results_summary()
        
        print("\n  💡 Next Steps:")
        remaining = [
            s for s in self.strategist.strategies
            if s not in self.strategist.execution_history
        ]
        for s in remaining[:3]:
            print(f"     • Try: {s}")
        
        print("=" * 60)
    
    def save_history(self, path: str = "agent_history.json") -> None:
        """Save optimization history to JSON."""
        with open(path, "w") as f:
            json.dump({
                "iterations": self.iteration,
                "best_score": self.best_score,
                "best_strategy": self.best_result.strategy_name if self.best_result else None,
                "target": self.target_score,
                "history": self.history,
            }, f, indent=2)
        print(f"History saved to {path}")
