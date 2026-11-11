import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self._tools_dict = {tool.name: tool for tool in tools}
        self.state: dict[str, Any] = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request, response: Response) -> Message:
        async with AsyncDial(
                base_url=self.endpoint,
                api_key=request.api_key,
                api_version=request.api_version,
        ) as client:
            chunks = await client.chat.completions.create(
                deployment_name=deployment_name,
                messages=self._prepare_messages(request.messages),
                tools=[tool.schema for tool in self.tools] or None,
                stream=True,
            )
            tool_call_index_map: dict[int, dict[str, Any]] = {}
            content = ''
            async for chunk in chunks:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    choice.append_content(delta.content)
                    content += delta.content
                if delta.custom_content and delta.custom_content.attachments:
                    for attachment in delta.custom_content.attachments:
                        choice.add_attachment(**attachment.model_dump(exclude_none=True))
                for tool_call_delta in delta.tool_calls or []:
                    tool_call = tool_call_index_map.setdefault(
                        tool_call_delta.index,
                        {'function': {'name': '', 'arguments': ''}},
                    )
                    if tool_call_delta.id:
                        tool_call['id'] = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call['type'] = tool_call_delta.type
                    if tool_call_delta.function:
                        tool_call['function']['name'] += tool_call_delta.function.name or ''
                        tool_call['function']['arguments'] += tool_call_delta.function.arguments or ''

        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content or None,
            tool_calls=[
                ToolCall.validate(tool_call_index_map[index])
                for index in sorted(tool_call_index_map)
            ] or None,
        )
        if assistant_message.tool_calls:
            tool_messages = await asyncio.gather(*[
                self._process_tool_call(
                    tool_call, choice, request.api_key,
                    request.headers.get('x-conversation-id', ''),
                )
                for tool_call in assistant_message.tool_calls
            ])
            self.state[TOOL_CALL_HISTORY_KEY].append(assistant_message.dict(exclude_none=True))
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)
            return await self.handle_request(deployment_name, choice, request, response)

        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        history = unpack_messages(messages, self.state[TOOL_CALL_HISTORY_KEY])
        history.insert(0, {'role': Role.SYSTEM.value, 'content': self.system_prompt})
        for message in history:
            print(json.dumps(message))
        return history

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[str, Any]:
        tool_name = tool_call.function.name
        tool = self._tools_dict[tool_name]
        stage = StageProcessor.open_stage(choice, tool_name)
        try:
            if tool.show_in_stage:
                stage.append_content('## Request arguments: \n')
                arguments = json.dumps(json.loads(tool_call.function.arguments), indent=2)
                stage.append_content(f'```json\n{arguments}\n```\n')
                stage.append_content('## Response: \n')
            message = await tool.execute(ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id,
            ))
            return message.dict(exclude_none=True)
        finally:
            StageProcessor.close_stage_safely(stage)
