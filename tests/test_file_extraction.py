import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aidial_client.types.chat.legacy.chat_completion import ToolCall
from aidial_sdk.chat_completion import Message, Role

from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


def make_params(arguments):
    return ToolCallParams(
        tool_call=ToolCall.validate({
            'id': 'call-1', 'type': 'function',
            'function': {'name': 'file_content_extraction_tool', 'arguments': json.dumps(arguments)},
        }),
        stage=MagicMock(), choice=MagicMock(), api_key='key', conversation_id='conversation',
    )


class ExtractorTests(unittest.TestCase):
    def extract(self, filename, content):
        with patch('task.utils.dial_file_conent_extractor.Dial') as dial:
            download = dial.return_value.files.download.return_value
            download.filename = filename
            download.get_content.return_value = content
            result = DialFileContentExtractor('http://localhost:8080', 'key').extract_text('files/test')
            dial.return_value._http_client.internal_http_client.close.assert_called_once()
            return result

    def test_csv_and_html(self):
        report = Path('tests/report.csv').read_bytes()
        csv = self.extract('report.csv', report)
        self.assertIn('1700', csv)
        self.assertIn('2025-10-05', csv)
        html = self.extract('page.html', b'<style>secret</style><p>Visible</p><script>bad()</script>')
        self.assertEqual(html, 'Visible')

    def test_pdf_and_unsupported_format(self):
        with patch('task.utils.dial_file_conent_extractor.pdfplumber.open') as open_pdf:
            open_pdf.return_value.__enter__.return_value.pages = [
                MagicMock(extract_text=MagicMock(return_value='Page 1')),
                MagicMock(extract_text=MagicMock(return_value=None)),
            ]
            self.assertEqual(self.extract('sample.pdf', b'pdf'), 'Page 1\n')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.extract('image.png', b'bytes')


class FileToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_paginated_manual_and_out_of_range(self):
        manual = Path('tests/microwave_manual.txt').read_text(encoding='utf-8')
        tool = FileContentExtractionTool('http://localhost:8080')
        with patch('task.tools.files.file_content_extraction_tool.DialFileContentExtractor') as extractor:
            extractor.return_value.extract_text.return_value = manual
            pages = []
            for page in (1, 2, 3):
                params = make_params({'file_url': 'files/manual.txt', 'page': page})
                result = await tool.execute(params)
                self.assertEqual(result.role, Role.TOOL)
                self.assertEqual(result.tool_call_id, 'call-1')
                self.assertIn(f'**Page #{page}. Total pages: 3**', result.content)
                pages.append(result.content.rsplit('\n\n**Page #', 1)[0])
                params.stage.append_content.assert_called()
            self.assertEqual(''.join(pages), manual)
            result = await tool.execute(make_params({'file_url': 'files/manual.txt', 'page': 4}))
            self.assertIn('Page 4 does not exist', result.content)

    async def test_errors_are_visible_and_schema_is_valid(self):
        tool = FileContentExtractionTool('http://localhost:8080')
        self.assertEqual(tool.schema['function']['name'], tool.name)
        params = make_params({'file_url': 'files/manual.txt', 'page': 0})
        with self.assertLogs('task.tools.base', level='ERROR'):
            result = await tool.execute(params)
        self.assertIn('Error:', result.content)
        params.stage.append_content.assert_called_with(result.content)

    async def test_tool_message_passthrough(self):
        tool = FileContentExtractionTool('http://localhost:8080')
        expected = Message(role=Role.TOOL, content='result', tool_call_id='call-1')
        with patch.object(tool, '_execute', return_value=expected):
            self.assertIs(await tool.execute(make_params({'file_url': 'files/a.txt'})), expected)


if __name__ == '__main__':
    unittest.main()
