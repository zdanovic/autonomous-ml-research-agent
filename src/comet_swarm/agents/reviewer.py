"""Reviewer agent for error analysis and code correction."""

from typing import Any

from comet_swarm.agents.base import BaseAgent
from comet_swarm.schemas import AgentRole, ErrorType, ExecutionResult


class ReviewerAgent(BaseAgent):
    """
    Reviewer agent that analyzes errors and suggests fixes.
    
    Responsibilities:
    - Classify execution errors
    - Suggest code fixes
    - Decide retry vs escalate
    - Review generated code before execution
    """

    role = AgentRole.REVIEWER

    def __init__(self, model: str | None = None):
        super().__init__(
            model=model,
            temperature=0.0,
            task_type="complex_bug_fix",  # Use premium model for bug fixing
        )

    def get_system_prompt(self, context: dict[str, Any]) -> str:
        """Build system prompt for error analysis."""
        error_type = context.get("error_type", "unknown")
        error_message = context.get("error_message", "No error message")
        code = context.get("code", "No code available")
        
        return f"""You are a senior ML engineer debugging code execution errors.

## Error Context
- **Error Type**: {error_type}
- **Error Message**: {error_message}

## Original Code
```python
{code[:2000]}
```

## Your Tasks
1. **Classify** the error (syntax, runtime, logic, memory, timeout, data_schema)
2. **Identify** the root cause
3. **Generate** a fixed version of the code
4. **Decide** if this can be auto-fixed or needs human review

## Output Format
```json
{{
    "error_classification": "syntax|runtime|logic|memory|timeout|data_schema",
    "root_cause": "Brief explanation of what went wrong",
    "can_auto_fix": true/false,
    "fix_description": "What the fix does",
    "fixed_code": "The corrected Python code"
}}
```

If the error cannot be auto-fixed, explain why in the root_cause."""

    async def analyze_error(
        self,
        execution_result: ExecutionResult,
        original_code: str,
    ) -> dict[str, Any]:
        """
        Analyze execution error and suggest fix.
        
        Args:
            execution_result: Result from code execution
            original_code: The code that failed
        
        Returns:
            Analysis with classification, cause, and potential fix
        """
        context = {
            "error_type": execution_result.error_type.value if execution_result.error_type else "unknown",
            "error_message": execution_result.error_message or execution_result.stderr[:500],
            "code": original_code,
        }
        
        user_message = f"""The code execution failed with the following error:

**Error Type**: {context['error_type']}
**Error Message**: 
{context['error_message']}

**Traceback**:
{execution_result.error_traceback or 'No traceback available'}

Analyze this error and provide a fix."""

        response = await self.invoke(user_message, context)
        
        # Parse response
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                
                # Map string classification to enum
                error_map = {
                    "syntax": ErrorType.SYNTAX,
                    "runtime": ErrorType.RUNTIME,
                    "logic": ErrorType.LOGIC,
                    "memory": ErrorType.MEMORY,
                    "timeout": ErrorType.TIMEOUT,
                    "data_schema": ErrorType.DATA_SCHEMA,
                }
                result["error_type_enum"] = error_map.get(
                    result.get("error_classification", "unknown"),
                    ErrorType.UNKNOWN,
                )
                return result
            except json.JSONDecodeError:
                pass
        
        return {
            "error_classification": "unknown",
            "root_cause": "Failed to analyze error",
            "can_auto_fix": False,
            "fix_description": None,
            "fixed_code": None,
            "error_type_enum": ErrorType.UNKNOWN,
        }

    async def review_code(
        self,
        code: str,
        purpose: str,
    ) -> dict[str, Any]:
        """
        Review code before execution for potential issues.
        
        Args:
            code: Code to review
            purpose: What the code is supposed to do
        
        Returns:
            Review with issues and suggestions
        """
        context = {"code": code}
        
        user_message = f"""Review this code before execution:

**Purpose**: {purpose}

```python
{code}
```

Check for:
1. Syntax errors
2. Missing imports
3. Hardcoded paths that should be variables
4. Potential memory issues (loading huge data without chunking)
5. Missing error handling

Respond in JSON:
```json
{{
    "is_safe": true/false,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1"],
    "revised_code": "improved code if needed, or null"
}}
```"""

        response = await self.invoke(user_message, context)
        
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return {
            "is_safe": True,
            "issues": [],
            "suggestions": [],
            "revised_code": None,
        }

    def classify_error_heuristic(self, error_message: str, stderr: str) -> ErrorType:
        """
        Quick heuristic error classification without LLM.
        
        Args:
            error_message: Error message
            stderr: Standard error output
        
        Returns:
            ErrorType classification
        """
        combined = f"{error_message} {stderr}".lower()
        
        if any(kw in combined for kw in ["syntaxerror", "invalid syntax", "unexpected eof"]):
            return ErrorType.SYNTAX
        
        if any(kw in combined for kw in ["killed", "oom", "out of memory", "memoryerror"]):
            return ErrorType.MEMORY
        
        if any(kw in combined for kw in ["timeout", "timed out", "deadline exceeded"]):
            return ErrorType.TIMEOUT
        
        if any(kw in combined for kw in ["keyerror", "column", "schema", "not found"]):
            return ErrorType.DATA_SCHEMA
        
        if any(kw in combined for kw in ["typeerror", "valueerror", "attributeerror"]):
            return ErrorType.RUNTIME
        
        return ErrorType.UNKNOWN

    async def should_retry(
        self,
        error_analysis: dict[str, Any],
        retry_count: int,
        max_retries: int = 3,
    ) -> tuple[bool, str]:
        """
        Decide whether to retry with a fix or escalate.
        
        Args:
            error_analysis: Result from analyze_error
            retry_count: Current retry attempt
            max_retries: Maximum allowed retries
        
        Returns:
            Tuple of (should_retry, reason)
        """
        # Hard limits
        if retry_count >= max_retries:
            return False, f"Max retries ({max_retries}) exceeded"
        
        # Non-retryable errors
        error_type = error_analysis.get("error_type_enum", ErrorType.UNKNOWN)
        if error_type in (ErrorType.MEMORY, ErrorType.TIMEOUT):
            return False, f"{error_type.value} errors require architectural changes"
        
        # Check if we have a fix
        if not error_analysis.get("can_auto_fix", False):
            return False, error_analysis.get("root_cause", "Cannot auto-fix")
        
        if not error_analysis.get("fixed_code"):
            return False, "No fixed code provided"
        
        return True, "Retrying with fixed code"
