"""Activity #1: Simple LangGraph Agent that uses A2A protocol.

This creates a LangGraph that wraps A2A client calls, demonstrating how to use
the A2A protocol within a LangGraph workflow.
"""
import logging
from typing import Annotated, TypedDict, List, Any
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgentState(TypedDict):
    """State for the simple A2A client agent."""
    messages: Annotated[List, add_messages]
    task_id: str | None
    context_id: str | None


class SimpleA2AAgent:
    """Simple LangGraph agent that uses A2A protocol."""

    def __init__(self, base_url: str = 'http://localhost:10000'):
        self.base_url = base_url
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self.agent_card = None
        self.client = None

    async def initialize(self):
        """Initialize A2A client."""
        logger.info(f"Fetching agent card from {self.base_url}")
        
        resolver = A2ACardResolver(
            httpx_client=self.httpx_client,
            base_url=self.base_url,
        )
        
        self.agent_card = await resolver.get_agent_card()
        logger.info(f"✅ Agent card fetched: {self.agent_card.name}")
        
        if self.agent_card.skills:
            logger.info(f"   Skills: {', '.join([s.name for s in self.agent_card.skills])}")
        
        self.client = A2AClient(
            httpx_client=self.httpx_client,
            agent_card=self.agent_card
        )

    async def close(self):
        await self.httpx_client.aclose()

    async def _call_a2a_server(self, state: SimpleAgentState) -> SimpleAgentState:
        """LangGraph node that calls A2A server."""
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            return state
        
        user_query = last_message.content
        
        # Prepare message payload
        payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': user_query}],
                'message_id': uuid4().hex,
            },
        }
        
        # Add task/context IDs for multi-turn (if available)
        if state.get("task_id") and state.get("context_id"):
            payload['message']['task_id'] = state["task_id"]
            payload['message']['context_id'] = state["context_id"]
            logger.info(f"Continuing conversation (task_id={state['task_id'][:8]}...)")
        else:
            logger.info("Starting new conversation")
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload)
        )
        
        # Send message
        logger.info(f"Sending: {user_query[:60]}...")
        response = await self.client.send_message(request)
        
        # Extract response as JSON
        response_dict = response.model_dump(mode='json', exclude_none=True)
        
        # Check for error
        if 'error' in response_dict:
            error_msg = response_dict['error'].get('message', 'Unknown error')
            logger.error(f"Error from server: {error_msg}")
            
            # If task completed, retry without task_id/context_id
            if 'terminal state' in error_msg.lower():
                logger.info("Task completed, retrying with new conversation...")
                payload['message'].pop('task_id', None)
                payload['message'].pop('context_id', None)
                request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
                response = await self.client.send_message(request)
                response_dict = response.model_dump(mode='json', exclude_none=True)
                
                if 'error' in response_dict:
                    return {
                        "messages": [AIMessage(content=f"❌ Error: {response_dict['error'].get('message')}")],
                        "task_id": None,
                        "context_id": None,
                    }
        
        # Extract result
        if 'result' in response_dict:
            result = response_dict['result']
            task_id = result.get('id')
            context_id = result.get('context_id')
            
            # Extract text from artifacts
            response_text = ""
            if 'artifacts' in result:
                for artifact in result.get('artifacts', []):
                    for part in artifact.get('parts', []):
                        if 'text' in part:
                            response_text += part['text'] + "\n"
            
            # Try message parts if no artifacts
            if not response_text and 'message' in result:
                message = result['message']
                for part in message.get('parts', []):
                    if 'text' in part:
                        response_text += part['text'] + "\n"
            
            # Debug output
            if not response_text:
                logger.warning(f"No text found. Result keys: {list(result.keys())}")
                import json
                logger.warning(f"Full result:\n{json.dumps(result, indent=2)[:1000]}")
                response_text = f"Response received but no text.\nResult keys: {list(result.keys())}"
            
            logger.info(f"✅ Response received ({len(response_text)} chars)")
            
            return {
                "messages": [AIMessage(content=response_text.strip())],
                "task_id": task_id,
                "context_id": context_id,
            }
        
        return {
            "messages": [AIMessage(content=f"Unexpected response: {str(response_dict)[:200]}")],
            "task_id": None,
            "context_id": None,
        }

    def build_graph(self):
        """Build LangGraph."""
        graph = StateGraph(SimpleAgentState)
        graph.add_node("call_a2a", self._call_a2a_server)
        graph.set_entry_point("call_a2a")
        graph.add_edge("call_a2a", END)
        return graph.compile()


async def main():
    """Run Activity #1: LangGraph agent using A2A protocol."""
    agent = SimpleA2AAgent()
    
    try:
        await agent.initialize()
        graph = agent.build_graph()
        
        # Test queries
        queries = [
            "What are the latest developments in AI in 2025?",
            "Find recent papers on transformer architectures",
        ]
        
        state = {
            "messages": [],
            "task_id": None,
            "context_id": None,
        }
        
        print("\n" + "="*70)
        print("🤖 Activity #1: LangGraph Agent with A2A Protocol")
        print("="*70)
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 70)
            
            state["messages"].append(HumanMessage(content=query))
            result = await graph.ainvoke(state)
            
            if result.get("messages"):
                last_msg = result["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"\n💬 Response:\n{last_msg.content}\n")
            
            state = result  # Update for multi-turn
        
        print("="*70)
        print("✅ Activity #1 Complete!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("💡 Make sure server is running: uv run python -m app")
    finally:
        await agent.close()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
