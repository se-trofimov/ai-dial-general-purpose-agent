import asyncio
import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionTool(BaseTool):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return 'file_content_extraction_tool'

    @property
    def description(self) -> str:
        return (
            'Read text from a DIAL attachment URL: PDF (text only), TXT, CSV or HTML. '
            'Start at page 1. Large files are split into 10,000-character pages; '
            'read further pages when needed to answer. Use the exact attached file URL.'
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'file_url': {'type': 'string', 'description': 'URL of the attached DIAL file.'},
                'page': {'type': 'integer', 'minimum': 1, 'default': 1,
                         'description': 'For large documents each page has 10,000 characters.'},
            },
            'required': ['file_url'],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        file_url = arguments['file_url']
        page = arguments.get('page', 1)
        if not isinstance(file_url, str) or not file_url:
            raise ValueError('file_url must be a non-empty string')
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError('page must be a positive integer')

        stage = tool_call_params.stage
        stage.append_content(f'## Request arguments: \n**File URL**: {file_url}\n**Page**: {page}\n')
        stage.append_content('## Response: \n')
        extractor = DialFileContentExtractor(self.endpoint, tool_call_params.api_key)
        content = await asyncio.to_thread(extractor.extract_text, file_url)
        if not content.strip():
            content = 'Error: File content not found.'
        else:
            page_size = 10_000
            total_pages = (len(content) + page_size - 1) // page_size
            if page > total_pages:
                content = f'Error: Page {page} does not exist. Total pages: {total_pages}'
            elif total_pages > 1:
                start = (page - 1) * page_size
                content = f'{content[start:start + page_size]}\n\n**Page #{page}. Total pages: {total_pages}**'
        stage.append_content(f'```text\n{content}\n```\n')
        return content
