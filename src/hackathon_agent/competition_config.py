"""
Competition Configuration - Auto-detection layer for universal competition support.

This module automatically detects:
- Target column (present in train, absent in test)
- ID column (common naming patterns)
- Group column (for GroupKFold CV)
- Problem type (regression vs classification)
- Metric (based on problem type and target distribution)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

ProblemType = Literal["regression", "binary", "multiclass"]
Metric = Literal["rmsle", "rmse", "mae", "auc", "f1", "accuracy", "logloss"]


@dataclass
class CompetitionConfig:
    """Configuration for a competition, auto-detected or manually specified."""
    
    name: str
    target_col: str
    id_col: str | None = None
    group_col: str | None = None
    problem_type: ProblemType = "regression"
    metric: Metric = "rmse"
    
    # Data paths
    train_path: str | None = None
    test_path: str | None = None
    
    # Detected features
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    high_cardinality_cols: list[str] = field(default_factory=list)
    
    # Data characteristics
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    target_skew: float = 0.0
    has_missing: bool = False
    
    @classmethod
    def auto_detect(
        cls,
        train: pd.DataFrame,
        test: pd.DataFrame,
        name: str = "unknown",
    ) -> CompetitionConfig:
        """
        Automatically detect competition configuration from data.
        
        Args:
            train: Training dataframe
            test: Test dataframe
            name: Competition name for identification
            
        Returns:
            CompetitionConfig with auto-detected settings
        """
        print("\n  🔍 AUTO-DETECTING COMPETITION CONFIG...")
        
        # 1. Find target column (in train, not in test)
        target_col = cls._detect_target(train, test)
        print(f"  ✓ Target: {target_col}")
        
        # 2. Find ID column
        id_col = cls._detect_id_column(train, test)
        print(f"  ✓ ID: {id_col or 'None'}")
        
        # 3. Detect problem type
        problem_type = cls._detect_problem_type(train, target_col)
        print(f"  ✓ Problem: {problem_type}")
        
        # 4. Find group column (for GroupKFold)
        group_col = cls._detect_group_column(train, target_col, id_col)
        print(f"  ✓ Group: {group_col or 'None (will use KFold)'}")
        
        # 5. Select metric
        metric = cls._select_metric(train, target_col, problem_type)
        print(f"  ✓ Metric: {metric}")
        
        # 6. Categorize columns
        numeric_cols, categorical_cols, high_cardinality = cls._categorize_columns(
            train, target_col, id_col
        )
        
        # 7. Data characteristics
        target_skew = float(train[target_col].skew()) if problem_type == "regression" else 0.0
        has_missing = train.isna().any().any() or test.isna().any().any()
        
        config = cls(
            name=name,
            target_col=target_col,
            id_col=id_col,
            group_col=group_col,
            problem_type=problem_type,
            metric=metric,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            high_cardinality_cols=high_cardinality,
            n_train=len(train),
            n_test=len(test),
            n_features=len(numeric_cols) + len(categorical_cols),
            target_skew=target_skew,
            has_missing=has_missing,
        )
        
        print(f"\n  📊 Dataset: {config.n_train} train, {config.n_test} test, {config.n_features} features")
        if config.target_skew > 1.0:
            print(f"  ⚠️ Target is skewed ({config.target_skew:.2f}) - consider log transform")
        
        return config
    
    @staticmethod
    def _detect_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
        """Find target column: present in train, absent in test."""
        train_cols = set(train.columns)
        test_cols = set(test.columns)
        
        # Columns only in train
        train_only = train_cols - test_cols
        
        # Common target names
        common_names = ["target", "label", "y", "class", "outcome"]
        
        # Priority 1: Common target name in train_only
        for name in common_names:
            if name in train_only:
                return name
            # Case-insensitive check
            for col in train_only:
                if col.lower() == name:
                    return col
        
        # Priority 2: Any column in train_only that's numeric
        for col in train_only:
            if pd.api.types.is_numeric_dtype(train[col]):
                return col
        
        # Priority 3: First column in train_only
        if train_only:
            return list(train_only)[0]
        
        # Fallback: look for 'target' anywhere
        for col in train.columns:
            if "target" in col.lower():
                return col
        
        raise ValueError("Could not auto-detect target column. Please specify manually.")
    
    @staticmethod
    def _detect_id_column(train: pd.DataFrame, test: pd.DataFrame) -> str | None:
        """Find ID column based on naming patterns."""
        common_patterns = ["id", "row_id", "data_id", "sample_id", "index"]
        
        for col in train.columns:
            col_lower = col.lower()
            if col_lower in common_patterns or col_lower.endswith("_id"):
                # Verify it's in both train and test
                if col in test.columns:
                    return col
        
        return None
    
    @staticmethod
    def _detect_problem_type(train: pd.DataFrame, target_col: str) -> ProblemType:
        """Detect if regression or classification."""
        target = train[target_col]
        
        # Check if integer with few unique values
        n_unique = target.nunique()
        
        if n_unique == 2:
            return "binary"
        elif n_unique <= 20 and target.dtype in ["int64", "int32", "object", "category"]:
            return "multiclass"
        else:
            return "regression"
    
    @staticmethod
    def _detect_group_column(
        train: pd.DataFrame,
        target_col: str,
        id_col: str | None,
    ) -> str | None:
        """Find suitable grouping column for GroupKFold."""
        exclude = {target_col, id_col} if id_col else {target_col}
        
        # Priority 1: Semantic column names (country, region, location)
        priority_names = ["country", "region", "location", "group", "category", "cluster"]
        for col in train.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in priority_names) and col not in exclude:
                n_unique = train[col].nunique()
                # Must have at least 3 groups for meaningful GroupKFold
                if 3 <= n_unique <= 100:
                    return col
        
        # Priority 2: Look for categorical columns with good cardinality
        candidates = []
        
        for col in train.columns:
            if col in exclude:
                continue
                
            # Check if categorical or object type
            if train[col].dtype == "object" or train[col].dtype.name == "category":
                n_unique = train[col].nunique()
                # Good group column: 3-100 unique values (at least 3 for CV)
                if 3 <= n_unique <= 100:
                    # Score by how evenly distributed the groups are
                    value_counts = train[col].value_counts()
                    evenness = value_counts.min() / value_counts.max()
                    candidates.append((col, n_unique, evenness))
        
        if not candidates:
            return None
        
        # Sort by evenness (higher = better for CV)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0]
    
    @staticmethod
    def _select_metric(
        train: pd.DataFrame,
        target_col: str,
        problem_type: ProblemType,
    ) -> Metric:
        """Select appropriate metric based on problem type and target."""
        if problem_type == "binary":
            return "auc"
        elif problem_type == "multiclass":
            return "logloss"
        else:
            # Regression: check if target is always positive
            target = train[target_col]
            if (target > 0).all():
                # Positive target - RMSLE is often used
                return "rmsle"
            else:
                return "rmse"
    
    @staticmethod
    def _categorize_columns(
        train: pd.DataFrame,
        target_col: str,
        id_col: str | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Categorize columns into numeric, categorical, and high-cardinality."""
        exclude = {target_col}
        if id_col:
            exclude.add(id_col)
        
        numeric_cols = []
        categorical_cols = []
        high_cardinality = []
        
        for col in train.columns:
            if col in exclude:
                continue
            
            if pd.api.types.is_numeric_dtype(train[col]):
                numeric_cols.append(col)
            else:
                n_unique = train[col].nunique()
                if n_unique > 50:
                    high_cardinality.append(col)
                else:
                    categorical_cols.append(col)
        
        return numeric_cols, categorical_cols, high_cardinality
    
    def get_cv_strategy(self) -> str:
        """Return appropriate CV strategy."""
        if self.group_col:
            return "group_kfold"
        else:
            return "kfold"
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print(f"  📋 COMPETITION CONFIG: {self.name.upper()}")
        print("=" * 60)
        print(f"  Target: {self.target_col}")
        print(f"  Problem: {self.problem_type}")
        print(f"  Metric: {self.metric}")
        print(f"  CV: {self.get_cv_strategy()}")
        if self.group_col:
            print(f"  Group: {self.group_col}")
        print(f"  Features: {self.n_features} ({len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical)")
        if self.high_cardinality_cols:
            print(f"  High Cardinality: {', '.join(self.high_cardinality_cols[:3])}...")
        print("=" * 60)


def load_competition_data(
    competition: str,
    data_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data for a competition.
    
    Args:
        competition: Competition name (e.g., "solafune", "wundernn")
        data_dir: Base data directory
        
    Returns:
        Tuple of (train_df, test_df)
    """
    data_path = Path(data_dir)
    
    # Try competition-specific subdirectory first
    comp_path = data_path / competition
    if comp_path.exists():
        # Look for parquet or csv
        train_files = list(comp_path.glob("train.*"))
        test_files = list(comp_path.glob("test.*"))
    else:
        # Try root data directory
        train_files = list(data_path.glob("train.*"))
        test_files = list(data_path.glob("test.*"))
    
    if not train_files or not test_files:
        raise FileNotFoundError(f"Could not find train/test files for competition '{competition}'")
    
    train_file = train_files[0]
    test_file = test_files[0]
    
    # Load based on extension
    if train_file.suffix == ".parquet":
        train = pd.read_parquet(train_file)
        test = pd.read_parquet(test_file)
    elif train_file.suffix == ".csv":
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
    else:
        raise ValueError(f"Unsupported file format: {train_file.suffix}")
    
    print(f"  Loaded {competition}: train={len(train)}, test={len(test)}")
    
    return train, test
