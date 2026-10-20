"""Scripted-model integration tests: real graph, dispatcher and MCP HTTP SDK."""
import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from mcp.client.streamable_http import streamablehttp_client as sdk_transport
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_chain import LLMChain, LLMConfig
from openai_chain import OpenAIChain
from parallel_search import MCP_URL, USER_AGENT, WebSearchConfig


class ScriptedModel:
    def bind_tools(self, tools):
        self.names = [t['function']['name'] if isinstance(t, dict) else t.name for t in tools]
        return self

    async def ainvoke(self, messages):
        await asyncio.sleep(0)
        question = next(m.content for m in messages if isinstance(m, HumanMessage))
        replies = [m for m in messages if isinstance(m, ToolMessage)]
        if question == 'local':
            if not replies:
                return AIMessage(content='', tool_calls=[{'name': 'search_documents', 'args': {'query': 'local'}, 'id': 'local'}])
            return AIMessage(content=replies[-1].content)
        if len(replies) < 3:
            step = len(replies)
            name = 'web_fetch' if step == 1 else 'web_search'
            args = {'urls': ['https://example.org/source']} if step == 1 else {'objective': question, 'search_queries': [question]}
            return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': str(step)}])
        # Consume actual tool messages to produce the final answer.
        return AIMessage(content='Sources: ' + ' '.join(m.content for m in replies))


class DocumentStore:
    def as_retriever(self, search_kwargs):
        self.filter = search_kwargs['filter']
        return self

    def invoke(self, query):
        return [Document(page_content='Local evidence', metadata={'file_path': '/container/report.txt'})]


class WebRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.calls = []
        self.mode = 'normal'
        self.waiting = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.patches = [patch.dict(os.environ, {'MINIMA_ALLOW_WEB': 'true'}, clear=True),
                        patch('openai_chain.OpenAIChain._setup_llm', side_effect=ScriptedModel),
                        patch('openai_chain.OpenAIChain._setup_document_store', side_effect=DocumentStore),
                        patch('parallel_search.streamablehttp_client', side_effect=self.transport)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def transport(self, url, **kwargs):
        def factory(headers=None, timeout=None, auth=None):
            return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth,
                                     transport=httpx.MockTransport(self.handle))
        return sdk_transport(url, **kwargs, httpx_client_factory=factory)

    async def handle(self, request):
        self.requests.append(request)
        payload = json.loads(request.content)
        method = payload['method']
        if 'id' not in payload:
            return httpx.Response(202)
        if method == 'initialize':
            result = {'protocolVersion': payload['params']['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'test-search', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': name, 'inputSchema': {'type': 'object'}} for name in ['web_search', 'web_fetch']]}
        elif method == 'tools/call':
            self.calls.append(payload['params'])
            if self.mode == 'http_error':
                return httpx.Response(401, text='test authorization rejected')
            if self.mode == 'hang':
                self.waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            result = {'isError': self.mode == 'tool_error', 'content': [{'type': 'text', 'text': json.dumps({
                'results': [{'url': 'https://example.org/source', 'title': 'A source', 'excerpts': ['Fresh evidence']}]
            })}]}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': payload['id'], 'result': result})

    def chain(self):
        return LLMChain(LLMConfig(llm_base_url='http://test-llm/v1', llm_model='scripted-model'))

    async def test_factory_graph_search_fetch_followup_and_sources(self):
        chain = self.chain()
        result = await chain.ainvoke('current facts', user_id='alice')
        self.assertIn('Fresh evidence', result['answer'])
        self.assertEqual(result['links'], {'https://example.org/source'})
        self.assertEqual([c['name'] for c in self.calls], ['web_search', 'web_fetch', 'web_search'])
        ids = {c['arguments']['session_id'] for c in self.calls}
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(next(iter(ids))), 36)
        self.assertEqual(chain.chain.llm.names, ['search_documents', 'web_search', 'web_fetch'])
        for request in self.requests:
            self.assertEqual(str(request.url), MCP_URL)
            self.assertEqual(request.headers['user-agent'], USER_AGENT)
            self.assertNotIn('authorization', request.headers)
        self.assertEqual({json.loads(r.content)['method'] for r in self.requests},
                         {'initialize', 'notifications/initialized', 'tools/list', 'tools/call'})

    async def test_concurrent_runs_same_chain_and_new_run_identity(self):
        chain = self.chain()
        await asyncio.gather(chain.ainvoke('conversation A'), chain.ainvoke('conversation B'))
        searches = [c['arguments'] for c in self.calls if c['name'] == 'web_search']
        by_run = {q: {c['session_id'] for c in searches if c['objective'] == q}
                  for q in ['conversation A', 'conversation B']}
        self.assertTrue(all(len(ids) == 1 for ids in by_run.values()))
        self.assertNotEqual(by_run['conversation A'], by_run['conversation B'])
        old_ids = {c['arguments']['session_id'] for c in self.calls}
        self.calls.clear()
        await chain.ainvoke('conversation A')
        self.assertTrue(old_ids.isdisjoint({c['arguments']['session_id'] for c in self.calls}))

    async def test_permissions_disabled_and_explicit_provider_choice(self):
        for env in [{'MINIMA_ALLOW_WEB': 'false'}, {'MINIMA_ALLOW_WEB': 'true', 'MINIMA_WEB_SEARCH': 'none'}, {}]:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                chain = self.chain()
                result = await chain.ainvoke('even a model requesting web cannot bypass permission')
                self.assertEqual(chain.chain.llm.names, ['search_documents'])
                self.assertIn('Web access is disabled', result['answer'])
                self.assertEqual(self.requests, [])
        with patch.dict(os.environ, {'MINIMA_WEB_SEARCH': 'unknown'}, clear=True):
            with self.assertRaises(ValueError):
                self.chain()

    async def test_configured_credentials_no_anonymous_fallback(self):
        with patch.dict(os.environ, {'PARALLEL_API_KEY': 'explicit-test-key'}):
            chain = self.chain()
        self.mode = 'http_error'
        result = await chain.ainvoke('auth failure')
        self.assertIn('Web request failed', result['answer'])
        self.assertNotIn('explicit-test-key', result['answer'])
        self.assertTrue(self.requests)
        self.assertTrue(all(r.headers.get('authorization') == 'Bearer explicit-test-key' for r in self.requests))
        # No transport retry and no switch to an anonymous provider.
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len({c['arguments']['session_id'] for c in self.calls}), 1)

    async def test_tool_error_keeps_run_identity(self):
        self.mode = 'tool_error'
        result = await self.chain().ainvoke('error')
        self.assertIn('Web request failed', result['answer'])
        self.assertEqual(result['links'], set())
        self.assertEqual(len({c['arguments']['session_id'] for c in self.calls}), 1)

    async def test_timeout_closes_request_and_returns_model_error(self):
        chain = self.chain()
        chain.chain.web_search.config.timeout = 0.05
        self.mode = 'hang'
        result = await chain.ainvoke('timeout')
        self.assertIn('Web request timed out', result['answer'])
        self.assertTrue(self.cancelled.is_set())
        self.assertEqual(len({c['arguments']['session_id'] for c in self.calls}), 1)

    async def test_cancellation_propagates_and_closes_request(self):
        self.mode = 'hang'
        task = asyncio.create_task(self.chain().ainvoke('cancel'))
        await asyncio.wait_for(self.waiting.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.cancelled.is_set())
        self.assertEqual(len(self.calls), 1)

    async def test_local_document_search_stays_filtered_and_links_work(self):
        chain = self.chain()
        chain.chain.localConfig.CONTAINER_PATH = '/container/'
        chain.chain.localConfig.LOCAL_FILES_PATH = '/local/'
        result = await chain.ainvoke('local', user_id='alice')
        self.assertIn('Local evidence', result['answer'])
        self.assertEqual(result['links'], {'file:///local/report.txt'})
        self.assertEqual(chain.chain.document_store.filter.must[0].match.value, 'alice')
        self.assertEqual(self.requests, [])

    async def test_invalid_urls_and_session_override_never_reach_network(self):
        chain = self.chain().chain
        for name, args in [('web_fetch', {'urls': ['file:///private']}),
                           ('web_fetch', {'urls': ['https://user:secret@example.org']}),
                           ('web_search', {'objective': 'q', 'search_queries': ['q'], 'session_id': 'model supplied'})]:
            msg = AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': 'bad'}])
            result = await chain._execute_tools({'messages': [msg], 'web_session_id': 'trusted'})
            self.assertIn('Invalid web tool arguments', result['messages'][0].content)
        self.assertEqual(self.requests, [])

    def test_configuration_defaults_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            config = WebSearchConfig()
        self.assertEqual(config.provider, 'parallel')
        self.assertFalse(config.allowed)
        self.assertEqual(config.api_key, '')
        for timeout in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                WebSearchConfig(timeout=timeout)


if __name__ == '__main__':
    unittest.main()
