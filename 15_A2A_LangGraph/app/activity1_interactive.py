"""Interactive Activity #1 client - type queries interactively."""
import asyncio
import logging
from uuid import uuid4
from typing import Annotated, TypedDict, List, Any

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgentState(TypedDict):
    """State for the A2A client agent."""
    messages: Annotated[List, add_messages]
    task_id: str | None
    context_id: str | None


class SimpleA2AAgent:
    """Simple LangGraph agent using A2A protocol."""

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
        logger.info(f"✅ Agent: {self.agent_card.name}")
        
        self.client = A2AClient(
            httpx_client=self.httpx_client,
            agent_card=self.agent_card
        )

    async def close(self):
        await self.httpx_client.aclose()

    async def _call_a2a_server(self, state: SimpleAgentState) -> SimpleAgentState:
        """Call A2A server."""
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            return state
        
        user_query = last_message.content
        
        # Prepare payload
        payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': user_query}],
                'message_id': uuid4().hex,
            },
        }
        
        # Add task/context for multi-turn
        if state.get("task_id") and state.get("context_id"):
            payload['message']['task_id'] = state["task_id"]
            payload['message']['context_id'] = state["context_id"]
        
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        
        # Send
        print(f"\n🔄 Sending to A2A server...")
        response = await self.client.send_message(request)
        response_dict = response.model_dump(mode='json', exclude_none=True)
        
        # Check error
        if 'error' in response_dict:
            error_msg = response_dict['error'].get('message', 'Unknown error')
            
            # Retry if task completed
            if 'terminal state' in error_msg.lower():
                logger.info("Task completed, starting new conversation...")
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
            for artifact in result.get('artifacts', []):
                for part in artifact.get('parts', []):
                    if 'text' in part:
                        response_text += part['text'] + "\n"
            
            # If no text in artifacts, try message parts
            if not response_text and 'message' in result:
                message = result['message']
                for part in message.get('parts', []):
                    if 'text' in part:
                        response_text += part['text'] + "\n"
            
            # Debug: show what we got
            if not response_text:
                logger.warning(f"No text found. Result keys: {list(result.keys())}")
                if 'artifacts' in result:
                    logger.warning(f"Artifacts: {result['artifacts']}")
                if 'message' in result:
                    logger.warning(f"Message: {result.get('message', {}).get('parts', [])}")
                response_text = f"Response received but no text content found.\nResult keys: {list(result.keys())}"
            
            return {
                "messages": [AIMessage(content=response_text.strip())],
                "task_id": task_id,
                "context_id": context_id,
            }
        
        return {
            "messages": [AIMessage(content="Unexpected response format")],
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


async def interactive_session():
    """Run interactive session."""
    agent = SimpleA2AAgent()
    
    try:
        await agent.initialize()
        graph = agent.build_graph()
        
        print("\n" + "="*60)
        print("🤖 Interactive A2A Agent")
        print("="*60)
        print("Type your queries. Type 'quit' or 'exit' to stop.")
        print("="*60 + "\n")
        
        state = {"messages": [], "task_id": None, "context_id": None}
        
        while True:
            try:
                query = input("\n💬 Your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                state["messages"].append(HumanMessage(content=query))
                result = await graph.ainvoke(state)
                
                # Display response
                if result.get("messages"):
                    last_msg = result["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        print(f"\n🤖 Response:\n{last_msg.content}")
                
                state = result  # Update for multi-turn
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Error: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Failed to start: {e}")
        print("💡 Make sure server is running: uv run python -m app")
    finally:
        await agent.close()


if __name__ == '__main__':
    asyncio.run(interactive_session())
