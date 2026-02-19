"""Base agent class with LLM integration and Opik tracing."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from comet_swarm.config import get_settings
from comet_swarm.integrations.opik import get_tracer
from comet_swarm.schemas import AgentRole


class BaseAgent(ABC):
    """Base class for all agents in the swarm."""

    role: AgentRole
    system_prompt: str

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        task_type: str = "code_generation",
    ):
        """
        Initialize base agent.
        
        Args:
            model: LLM model name (auto-selected if None based on task)
            temperature: Sampling temperature
            task_type: Task type for model tier selection
        """
        settings = get_settings()
        self.tracer = get_tracer()
        
        # Select model based on task type and budget
        if model is None:
            model = settings.get_model_for_task(task_type)
        
        self.model_name = model
        self.task_type = task_type
        
        # Initialize LLM based on provider
        api_key = self._get_api_key(settings, model)
        base_url = self._get_base_url(settings, model)
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    def _get_api_key(self, settings, model: str) -> str:
        """Get API key based on model/provider."""
        
        # Check if OpenRouter is configured and model uses OpenRouter format
        if settings.openrouter_api_key and "/" in model:
            return settings.openrouter_api_key.get_secret_value()
        
        # Direct provider keys
        if settings.openai_api_key and ("gpt" in model.lower() or "openai" in model.lower()):
            return settings.openai_api_key.get_secret_value()
        
        if settings.anthropic_api_key and "claude" in model.lower():
            return settings.anthropic_api_key.get_secret_value()
        
        # Fallback to OpenRouter if available
        if settings.openrouter_api_key:
            return settings.openrouter_api_key.get_secret_value()
        
        raise ValueError("No API key configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY")

    def _get_base_url(self, settings, model: str) -> str | None:
        """Get base URL for provider."""
        # OpenRouter format: provider/model (e.g., openai/gpt-4o)
        if settings.openrouter_api_key and "/" in model:
            return "https://openrouter.ai/api/v1"
        
        # Direct providers use default URLs
        return None

    @abstractmethod
    def get_system_prompt(self, context: dict[str, Any]) -> str:
        """Get system prompt with context."""
        pass

    async def invoke(
        self,
        user_message: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Invoke the agent with a message.
        
        Args:
            user_message: User's input message
            context: Additional context for the agent
        
        Returns:
            Agent's response string
        """
        context = context or {}
        
        # Build messages
        messages: list[BaseMessage] = [
            SystemMessage(content=self.get_system_prompt(context)),
            HumanMessage(content=user_message),
        ]
        
        # Invoke LLM
        response = await self.llm.ainvoke(messages)
        response_text = response.content
        
        # Calculate tokens (approximate)
        input_tokens = sum(len(m.content) // 4 for m in messages)
        output_tokens = len(response_text) // 4
        
        # Log to Opik
        self.tracer.log_agent_step(
            agent_role=self.role,
            input_prompt=user_message[:500],
            output_response=str(response_text)[:500],
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_type=self.task_type,
        )
        
        return str(response_text)

    def invoke_sync(
        self,
        user_message: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous version of invoke."""
        import asyncio
        return asyncio.run(self.invoke(user_message, context))


class ToolCallMixin:
    """Mixin for agents that can call tools."""

    def format_tools_for_prompt(self, tools: list[dict[str, str]]) -> str:
        """Format available tools for the system prompt."""
        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(
                f"- **{tool['name']}**: {tool['description']}\n"
                f"  Usage: `{tool['usage']}`"
            )
        return "\n".join(tool_descriptions)

    def parse_tool_call(self, response: str) -> dict[str, Any] | None:
        """Parse tool call from agent response."""
        import json
        import re
        
        # Look for JSON code block
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for inline JSON
        try:
            # Find JSON object
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return None
