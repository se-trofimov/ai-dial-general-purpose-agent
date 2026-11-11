import copy
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from aidial_client import AsyncDial
from aidial_client.types.chat.response import ChatCompletionChunk
from aidial_sdk.chat_completion import Message, Request, Role
from starlette.requests import Request as HttpRequest

from task.agent import GeneralPurposeAgent
from task.app import app
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages


async def stream_chunks(deltas):
    for delta in deltas:
        yield ChatCompletionChunk.model_validate({
            'id': 'completion',
            'object': 'chat.completion.chunk',
            'created': 0,
            'choices': [{'index': 0, 'delta': delta}],
        })
    yield ChatCompletionChunk.model_validate({
        'id': 'completion', 'object': 'chat.completion.chunk',
        'created': 0, 'choices': [],
    })


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = GeneralPurposeAgent('http://localhost:8080', 'System rules', [])
        self.choice = MagicMock()
        self.request = Request(
            deployment_id='general-purpose-agent',
            original_request=HttpRequest({'type': 'http', 'headers': []}),
            api_key_secret='per-request-key',
            api_version='2024-02-01',
            headers={'x-conversation-id': 'conversation'},
            messages=[Message(role=Role.USER, content='Hello')],
        )
        self.client = MagicMock()
        self.client.chat.completions.create = AsyncMock()
        self.client.__aenter__ = AsyncMock(return_value=self.client)
        self.client.__aexit__ = AsyncMock(return_value=False)
        self.client_patch = patch('task.agent.AsyncDial', return_value=self.client)
        self.dial = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.print_patch = patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    async def test_streaming_without_tools_and_attachment(self):
        self.client.chat.completions.create.return_value = stream_chunks([
            {'content': 'Hel'},
            {'content': 'lo', 'custom_content': {'attachments': [
                {'type': 'image/png', 'title': 'Image', 'url': 'http://localhost/image.png'},
            ]}},
        ])
        message = await self.agent.handle_request('gpt-4o', self.choice, self.request, MagicMock())
        self.assertEqual(message.content, 'Hello')
        self.assertEqual([call.args[0] for call in self.choice.append_content.call_args_list], ['Hel', 'lo'])
        self.choice.add_attachment.assert_called_once_with(
            type='image/png', title='Image', url='http://localhost/image.png',
        )
        self.choice.set_state.assert_called_once_with({TOOL_CALL_HISTORY_KEY: []})
        self.dial.assert_called_once_with(
            base_url='http://localhost:8080', api_key='per-request-key', api_version='2024-02-01',
        )
        params = self.client.chat.completions.create.call_args.kwargs
        self.assertIsNone(params['tools'])
        self.assertEqual(params['messages'][0], {'role': 'system', 'content': 'System rules'})

    async def test_interleaved_tool_arguments_and_followup(self):
        self.client.chat.completions.create.side_effect = [
            stream_chunks([
                {'tool_calls': [
                    {'index': 0, 'id': 'first', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"q":'}},
                    {'index': 1, 'id': 'second', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"q":'}},
                ]},
                {'tool_calls': [{'index': 1, 'function': {'arguments': '2}'}}]},
                {'tool_calls': [{'index': 0, 'function': {'arguments': '1}'}}]},
            ]),
            stream_chunks([{'content': 'Done'}]),
        ]

        async def process(tool_call, choice, api_key, conversation_id):
            self.assertEqual(api_key, 'per-request-key')
            self.assertEqual(conversation_id, 'conversation')
            return {'role': 'tool', 'tool_call_id': tool_call.id, 'content': tool_call.function.arguments}

        with patch.object(self.agent, '_process_tool_call', side_effect=process) as execute:
            message = await self.agent.handle_request('gpt-4o', self.choice, self.request, MagicMock())
        self.assertEqual(message.content, 'Done')
        self.assertEqual(execute.call_count, 2)
        history = self.client.chat.completions.create.call_args.kwargs['messages']
        calls = history[2]['tool_calls']
        self.assertEqual([call['function']['arguments'] for call in calls], ['{"q":1}', '{"q":2}'])
        self.assertEqual([entry['tool_call_id'] for entry in history[3:]], ['first', 'second'])
        self.choice.set_state.assert_called_once()

    async def test_tool_failure_closes_stage_and_propagates(self):
        tool = MagicMock()
        tool.show_in_stage = True
        tool.execute = AsyncMock(side_effect=RuntimeError('Tool failed'))
        self.agent._tools_dict['lookup'] = tool
        from aidial_client.types.chat.legacy.chat_completion import ToolCall
        tool_call = ToolCall.validate({
            'id': 'first', 'type': 'function',
            'function': {'name': 'lookup', 'arguments': '{}'},
        })
        with self.assertRaisesRegex(RuntimeError, 'Tool failed'):
            await self.agent._process_tool_call(tool_call, self.choice, 'key', 'conversation')
        self.choice.create_stage.return_value.close.assert_called_once()

    async def test_http_endpoint_streams_reply(self):
        self.client.chat.completions.create.return_value = stream_chunks([
            {'content': 'Hello'}, {'content': ' world'},
        ])
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url='http://localhost',
        ) as client:
            result = await client.post(
                '/openai/deployments/general-purpose-agent/chat/completions',
                headers={'api-key': 'per-request-key'},
                json={'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True},
            )
        self.assertEqual(result.status_code, 200, result.text)
        events = [json.loads(line[6:]) for line in result.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        content = ''.join(choice['delta'].get('content') or ''
                          for event in events for choice in event.get('choices', []))
        self.assertEqual(content, 'Hello world')
        self.assertIn('data: [DONE]', result.text)

    async def test_real_sdk_stream_and_client_cleanup(self):
        def respond(request):
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=(
                'data: {"id":"test","object":"chat.completion.chunk","created":0,'
                '"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
                'data: [DONE]\n\n'
            ))

        transport_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch('aidial_client._http_client._async.AsyncHTTPClient._create_internal_client',
                   return_value=transport_client):
            client = AsyncDial(base_url='http://localhost:8080', api_key='test')
        self.dial.return_value = client
        message = await self.agent.handle_request('gpt-4o', self.choice, self.request, MagicMock())
        self.assertEqual(message.content, 'Hello')
        self.assertTrue(transport_client.is_closed)


class HistoryTests(unittest.TestCase):
    def test_plain_assistant_messages_are_preserved(self):
        messages = [
            Message(role=Role.USER, content='My name is Alex'),
            Message(role=Role.ASSISTANT, content='Hello Alex'),
            Message(role=Role.USER, content='What is my name?'),
        ]
        self.assertEqual([msg['content'] for msg in unpack_messages(messages, [])],
                         ['My name is Alex', 'Hello Alex', 'What is my name?'])

    def test_saved_tool_history_is_unpacked_without_mutation(self):
        saved_history = [
            {'role': 'assistant', 'tool_calls': [{'id': 'call', 'type': 'function',
              'function': {'name': 'lookup', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'call', 'content': 'Result'},
        ]
        messages = [Message(role=Role.ASSISTANT, content='Answer',
                            custom_content={'state': {TOOL_CALL_HISTORY_KEY: saved_history}})]
        current_history = [{'role': 'assistant', 'content': 'Current', 'custom_content': {'state': {}}}]
        original = copy.deepcopy(current_history)
        result = unpack_messages(messages, current_history)
        self.assertEqual([msg['role'] for msg in result], ['assistant', 'tool', 'assistant', 'assistant'])
        self.assertNotIn('custom_content', result[-1])
        self.assertEqual(current_history, original)


if __name__ == '__main__':
    unittest.main()
