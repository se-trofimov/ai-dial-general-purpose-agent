import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from aidial_client.types.chat.legacy.chat_completion import ToolCall
from aidial_sdk.chat_completion import Role

from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool
from test_agent import stream_chunks


def params(request='How to clean?', conversation='conv'):
    return ToolCallParams(
        tool_call=ToolCall.validate({
            'id': 'call', 'type': 'function',
            'function': {'name': 'rag_document_search',
                         'arguments': json.dumps({'request': request, 'file_url': 'files/manual.txt'})},
        }),
        stage=MagicMock(), choice=MagicMock(), api_key='request-key', conversation_id=conversation,
    )


class RagRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tool = RagTool('http://localhost:8080', 'gpt-4o', DocumentCache())
        self.model = MagicMock()
        self.model.encode.side_effect = lambda texts: np.array(
            [[float('clean' in text), float('cook' in text)] for text in texts], dtype=np.float32,
        )
        self.tool._model = self.model
        self.tool.text_splitter.split_text = MagicMock(return_value=['clean tray', 'cook food'])

    def test_cache_rechecks_access_and_invalidates_changes(self):
        with patch('task.tools.rag.rag_tool.DialFileContentExtractor') as extractor:
            extract = extractor.return_value.extract_text
            extract.return_value = 'Original content'
            self.assertEqual(self.tool._retrieve('clean', 'files/manual.txt', 'key1', 'conv')[0], 'clean tray')
            self.tool._retrieve('cook', 'files/manual.txt', 'key2', 'conv')
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(self.tool.text_splitter.split_text.call_count, 1)
            extract.return_value = 'Changed content'
            self.tool._retrieve('clean', 'files/manual.txt', 'key2', 'conv')
            self.assertEqual(self.tool.text_splitter.split_text.call_count, 2)
            extract.side_effect = PermissionError('access denied')
            with self.assertRaises(PermissionError):
                self.tool._retrieve('clean', 'files/manual.txt', 'key3', 'conv')

    def test_missing_conversation_does_not_share_cache(self):
        with patch('task.tools.rag.rag_tool.DialFileContentExtractor') as extractor:
            extractor.return_value.extract_text.return_value = 'Content'
            self.tool._retrieve('clean', 'files/a.txt', 'key', '')
            self.tool._retrieve('clean', 'files/a.txt', 'key', '')
        self.assertEqual(self.tool.document_cache.size(), 0)
        self.assertEqual(self.tool.text_splitter.split_text.call_count, 2)

    def test_empty_document_is_an_error(self):
        with patch('task.tools.rag.rag_tool.DialFileContentExtractor') as extractor:
            extractor.return_value.extract_text.return_value = '  '
            with self.assertRaisesRegex(ValueError, 'content not found'):
                self.tool._retrieve('clean', 'files/a.txt', 'key', 'conv')

    def test_normalizes_line_endings_before_splitting_and_caching(self):
        with patch('task.tools.rag.rag_tool.DialFileContentExtractor') as extractor:
            extractor.return_value.extract_text.side_effect = ['one\r\ntwo\rthree', 'one\ntwo\nthree']
            self.tool._retrieve('clean', 'files/a.txt', 'key', 'conv')
            self.tool._retrieve('clean', 'files/a.txt', 'key', 'conv')
        self.tool.text_splitter.split_text.assert_called_once_with('one\ntwo\nthree')


class RagGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_generation_receives_selected_excerpts(self):
        tool = RagTool('http://localhost:8080', 'gpt-4o', DocumentCache())
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=stream_chunks([
            {'content': 'Wash '}, {'content': 'the tray.'},
        ]))
        call = params()
        with patch.object(tool, '_retrieve', return_value=['Wash the tray in warm water.']), \
                patch('task.tools.rag.rag_tool.AsyncDial', return_value=client):
            message = await tool.execute(call)
        self.assertEqual(message.role, Role.TOOL)
        self.assertEqual(message.content, 'Wash the tray.')
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args['deployment_name'], 'gpt-4o')
        data = json.loads(args['messages'][1]['content'])
        self.assertEqual(data['question'], 'How to clean?')
        self.assertEqual(data['document_excerpts'], ['Wash the tray in warm water.'])
        self.assertIn('untrusted', args['messages'][0]['content'])
        call.stage.append_content.assert_any_call('Wash ')
        call.stage.append_content.assert_any_call('the tray.')

    async def test_empty_generation_is_surfaced(self):
        tool = RagTool('http://localhost:8080', 'gpt-4o', DocumentCache())
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=stream_chunks([]))
        with patch.object(tool, '_retrieve', return_value=['excerpt']), \
                patch('task.tools.rag.rag_tool.AsyncDial', return_value=client), \
                self.assertLogs('task.tools.base', level='ERROR'):
            message = await tool.execute(params())
        self.assertIn('Error:', message.content)


if __name__ == '__main__':
    unittest.main()
