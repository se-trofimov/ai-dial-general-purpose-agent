from abc import ABC, abstractmethod
import logging
from typing import Any

from aidial_client.types.chat import ToolParam, FunctionParam
from aidial_sdk.chat_completion import Message, Role

from task.tools.models import ToolCallParams

logger = logging.getLogger(__name__)

class BaseTool(ABC):

    async def execute(self, tool_call_params: ToolCallParams) -> Message:
        try:
            result = await self._execute(tool_call_params)
            if isinstance(result, Message):
                return result
            content = result
        except Exception:
            logger.exception('Tool %s failed', tool_call_params.tool_call.function.name)
            content = 'Error: Tool execution failed. Check the server logs for details.'
            tool_call_params.stage.append_content(content)
        return Message(
            role=Role.TOOL,
            name=tool_call_params.tool_call.function.name,
            tool_call_id=tool_call_params.tool_call.id,
            content=content,
        )

    @abstractmethod
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        pass

    @property
    def show_in_stage(self) -> bool:
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def schema(self) -> ToolParam:
        """Provides tool schema according to DIAL specification."""
        return ToolParam(
            type="function",
            function=FunctionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters
            )
        )
