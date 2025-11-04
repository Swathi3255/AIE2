"""Debug script to inspect A2A response structure - use this to troubleshoot."""
import asyncio
import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


async def main():
    """Inspect response structure from A2A server."""
    base_url = 'http://localhost:10000'
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        # Initialize client
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        # Send a test message
        payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': 'What are the latest AI developments?'}],
                'message_id': uuid4().hex,
            },
        }
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))
        
        print("Sending request to A2A server...")
        response = await client.send_message(request)
        
        # Dump full response
        print("\n" + "="*80)
        print("FULL A2A RESPONSE STRUCTURE")
        print("="*80)
        response_dict = response.model_dump(mode='json', exclude_none=True)
        print(json.dumps(response_dict, indent=2))
        print("="*80)
        
        # Show where text is located
        if 'result' in response_dict:
            result = response_dict['result']
            print(f"\n✅ Result ID: {result.get('id')}")
            print(f"✅ Context ID: {result.get('context_id')}")
            
            if 'artifacts' in result:
                print(f"\n📦 Found {len(result['artifacts'])} artifact(s)")
                for i, artifact in enumerate(result['artifacts'], 1):
                    print(f"  Artifact {i}: {artifact.get('name')}")
                    for j, part in enumerate(artifact.get('parts', []), 1):
                        if 'text' in part:
                            text = part['text']
                            print(f"    Part {j} text ({len(text)} chars): {text[:100]}...")
        elif 'error' in response_dict:
            error = response_dict['error']
            print(f"\n❌ Error: {error.get('message')} (code: {error.get('code')})")


if __name__ == '__main__':
    asyncio.run(main())
