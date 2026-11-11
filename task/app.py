import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
# DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'claude-haiku-4-5')


class GeneralPurposeAgentApplication(ChatCompletion):

    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        #TODO:
        # 1. Create list of BaseTool
        # 2. Create MCPClient
        # 3. Get tools, iterate through them and add them to created list as MCPTool where the client will be created
        #    MCPClient and mcp_tool_model will be the tool itself (see what `mcp_client.get_tools` returns).
        # 4. Return created tool list
        raise NotImplementedError()

    async def _create_tools(self) -> list[BaseTool]:
        #TODO:
        # 1. Create list of BaseTool
        # ---
        # At the beginning this list can be empty. We will add here tools after they will be implemented
        # ---
        # 2. Add ImageGenerationTool with DIAL_ENDPOINT
        # 3. Add FileContentExtractionTool with DIAL_ENDPOINT
        # 4. Add RagTool with DIAL_ENDPOINT, DEPLOYMENT_NAME, and create DocumentCache (it has static method `create`)
        # 5. Add PythonCodeInterpreterTool with DIAL_ENDPOINT, `http://localhost:8050/mcp` mcp_url, tool_name is
        #    `execute_code`, more detailed about tools see in repository https://github.com/khshanovskyi/mcp-python-code-interpreter
        # 6. Extend tools with MCP tools from `http://localhost:8051/mcp` (use method `_get_mcp_tools`)
        return [FileContentExtractionTool(DIAL_ENDPOINT)]

    async def chat_completion(self, request: Request, response: Response) -> None:
        if not self.tools:
            self.tools = await self._create_tools()
        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )
            await agent.handle_request(DEPLOYMENT_NAME, choice, request, response)


app = DIALApp()
agent_app = GeneralPurposeAgentApplication()
app.add_chat_completion(deployment_name='general-purpose-agent', impl=agent_app)

if __name__ == '__main__':
    uvicorn.run(app, port=5030, host='0.0.0.0')
