"""E2B sandbox executor for safe code execution."""

from pathlib import Path
from typing import Any

from comet_swarm.config import get_settings
from comet_swarm.integrations.opik import traced
from comet_swarm.schemas import ErrorType, ExecutionResult, ExecutionStatus


class E2BExecutor:
    """
    Executor for running code in E2B sandboxed environments.
    
    Features:
    - Isolated execution environment
    - Timeout and memory limits
    - Automatic error classification
    - Artifact collection
    """

    def __init__(
        self,
        timeout_seconds: int | None = None,
        memory_mb: int | None = None,
        template: str | None = None,
    ):
        """
        Initialize E2B executor.
        
        Args:
            timeout_seconds: Execution timeout (default from settings)
            memory_mb: Memory limit in MB (default from settings)
            template: E2B template to use (default from settings)
        """
        settings = get_settings()
        
        self.timeout = timeout_seconds or settings.sandbox_timeout_seconds
        self.memory_mb = memory_mb or settings.sandbox_memory_mb
        self.template = template or settings.sandbox_template
        self.api_key = settings.e2b_api_key.get_secret_value()
        
        self._sandbox = None

    async def _create_sandbox(self):
        """Create E2B sandbox instance."""
        from e2b_code_interpreter import Sandbox
        
        self._sandbox = Sandbox(
            api_key=self.api_key,
            template=self.template,
            timeout=self.timeout,
        )
        return self._sandbox

    @traced(name="e2b_execute")
    async def execute(
        self,
        code: str,
        files: dict[str, str | bytes] | None = None,
        collect_artifacts: bool = True,
    ) -> ExecutionResult:
        """
        Execute code in E2B sandbox.
        
        Args:
            code: Python code to execute
            files: Optional files to upload (path -> content)
            collect_artifacts: Whether to download generated files
        
        Returns:
            ExecutionResult with status, output, and artifacts
        """
        try:
            from e2b_code_interpreter import Sandbox
            
            sandbox = Sandbox(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            
            # Upload files if provided
            if files:
                for path, content in files.items():
                    if isinstance(content, str):
                        content = content.encode()
                    sandbox.filesystem.write(path, content)
            
            # Execute code
            execution = sandbox.run_code(code, timeout=self.timeout)
            
            # Collect results
            stdout = ""
            stderr = ""
            artifacts: dict[str, Any] = {}
            
            # Process results
            for result in execution.results:
                if hasattr(result, "text"):
                    stdout += result.text + "\n"
            
            if execution.error:
                stderr = str(execution.error)
            
            # Collect artifacts if requested
            if collect_artifacts and execution.results:
                for result in execution.results:
                    if hasattr(result, "png"):
                        artifacts["chart.png"] = result.png
                    if hasattr(result, "data"):
                        artifacts["data"] = result.data
            
            # Determine status
            if execution.error:
                error_type = self._classify_error(stderr)
                status = ExecutionStatus.ERROR
                
                # Check for specific error types
                if "killed" in stderr.lower() or "oom" in stderr.lower():
                    status = ExecutionStatus.OOM
                    error_type = ErrorType.MEMORY
                elif "timeout" in stderr.lower():
                    status = ExecutionStatus.TIMEOUT
                    error_type = ErrorType.TIMEOUT
                
                return ExecutionResult(
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    error_type=error_type,
                    error_message=str(execution.error),
                    artifacts=artifacts,
                )
            
            # Close sandbox
            sandbox.close()
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=stdout,
                stderr=stderr,
                execution_time_seconds=0.0,  # E2B doesn't expose this directly
                artifacts=artifacts,
            )
            
        except ImportError:
            # E2B not installed, fall back to local execution
            return await self._execute_local(code)
        except TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_type=ErrorType.TIMEOUT,
                error_message=f"Execution timed out after {self.timeout}s",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_type=ErrorType.UNKNOWN,
                error_message=str(e),
                stderr=str(e),
            )

    async def _execute_local(self, code: str) -> ExecutionResult:
        """
        Fallback local execution (less secure).
        
        Only used when E2B is not available.
        """
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=min(self.timeout, 300),  # Cap at 5 minutes locally
            )
            
            if result.returncode == 0:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error_type=self._classify_error(result.stderr),
                    error_message=result.stderr[:500],
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_type=ErrorType.TIMEOUT,
                error_message="Local execution timed out",
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _classify_error(self, error_text: str) -> ErrorType:
        """Classify error type from error message."""
        error_lower = error_text.lower()
        
        if any(kw in error_lower for kw in ["syntaxerror", "invalid syntax"]):
            return ErrorType.SYNTAX
        if any(kw in error_lower for kw in ["memoryerror", "oom", "killed"]):
            return ErrorType.MEMORY
        if any(kw in error_lower for kw in ["timeout", "timed out"]):
            return ErrorType.TIMEOUT
        if any(kw in error_lower for kw in ["keyerror", "column not found"]):
            return ErrorType.DATA_SCHEMA
        if any(kw in error_lower for kw in ["typeerror", "valueerror", "attributeerror"]):
            return ErrorType.RUNTIME
        
        return ErrorType.UNKNOWN

    async def install_packages(self, packages: list[str]) -> bool:
        """
        Install packages in the sandbox.
        
        Args:
            packages: List of package names to install
        
        Returns:
            True if installation succeeded
        """
        pip_code = f"!pip install {' '.join(packages)}"
        result = await self.execute(pip_code)
        return result.status == ExecutionStatus.SUCCESS

    async def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
    ) -> bool:
        """
        Upload a file to the sandbox.
        
        Args:
            local_path: Local file path
            remote_path: Destination path in sandbox
        
        Returns:
            True if upload succeeded
        """
        try:
            from e2b_code_interpreter import Sandbox
            
            local_path = Path(local_path)
            content = local_path.read_bytes()
            
            sandbox = Sandbox(api_key=self.api_key)
            sandbox.filesystem.write(remote_path, content)
            sandbox.close()
            
            return True
        except Exception:
            return False

    async def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
    ) -> bool:
        """
        Download a file from the sandbox.
        
        Args:
            remote_path: Path in sandbox
            local_path: Local destination path
        
        Returns:
            True if download succeeded
        """
        try:
            from e2b_code_interpreter import Sandbox
            
            sandbox = Sandbox(api_key=self.api_key)
            content = sandbox.filesystem.read(remote_path)
            sandbox.close()
            
            Path(local_path).write_bytes(content)
            return True
        except Exception:
            return False
