"""Test runner for the LangGraph A2A Client Agent

This script demonstrates the LangGraph A2A client agent capabilities:
- Single-turn conversations
- Multi-turn conversations with context preservation
- Error handling
- Different query types (web search, academic search, RAG)

Usage:
    uv run python app/test_client_agent.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from app.client_agent import LangGraphA2AClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_single_turn_conversation(client: LangGraphA2AClient) -> None:
    """Test a simple single-turn conversation."""
    logger.info("=" * 60)
    logger.info("TEST 1: Single-turn conversation")
    logger.info("=" * 60)
    
    query = "Hello! Can you tell me what you can help me with?"
    
    try:
        result = await client.send_query(query)
        
        if result.get("error_message"):
            logger.error(f"Error in single-turn test: {result['error_message']}")
            if result.get("user_friendly_error"):
                print(f"❌ Error: {result['user_friendly_error']}")
        else:
            print(f"✅ Query: {query}")
            print(f"✅ Response: {result.get('response_content', 'No response content')}")
            print(f"✅ Task ID: {result.get('task_id')}")
            print(f"✅ Context ID: {result.get('context_id')}")
            
            return result.get('task_id'), result.get('context_id')
            
    except Exception as e:
        logger.error(f"Exception in single-turn test: {e}")
        print(f"❌ Exception: {str(e)}")
    
    return None, None


async def test_multi_turn_conversation(client: LangGraphA2AClient, task_id: str, context_id: str) -> None:
    """Test multi-turn conversation with context preservation."""
    logger.info("=" * 60)
    logger.info("TEST 2: Multi-turn conversation")
    logger.info("=" * 60)
    
    if not task_id or not context_id:
        logger.warning("Skipping multi-turn test - no context from previous conversation")
        print("⚠️ Skipping multi-turn test - no context available")
        return
    
    follow_up_query = "Can you elaborate on your capabilities?"
    
    try:
        # Start a new task but keep context_id for continuity (more realistic multi-turn)
        result = await client.send_query(
            user_query=follow_up_query,
            task_id=None,  # Don't reuse completed task_id
            context_id=context_id
        )
        
        if result.get("error_message"):
            logger.error(f"Error in multi-turn test: {result['error_message']}")
            if result.get("user_friendly_error"):
                print(f"❌ Error: {result['user_friendly_error']}")
        else:
            print(f"✅ Follow-up Query: {follow_up_query}")
            print(f"✅ Response: {result.get('response_content', 'No response content')}")
            print(f"✅ New task with preserved context: {result.get('task_id')}")
            
    except Exception as e:
        logger.error(f"Exception in multi-turn test: {e}")
        print(f"❌ Exception: {str(e)}")


async def test_web_search_capability(client: LangGraphA2AClient) -> None:
    """Test web search tool capability."""
    logger.info("=" * 60)
    logger.info("TEST 3: Web search capability")
    logger.info("=" * 60)
    
    query = "What are the latest developments in artificial intelligence in 2024?"
    
    try:
        result = await client.send_query(query)
        
        if result.get("error_message"):
            logger.error(f"Error in web search test: {result['error_message']}")
            if result.get("user_friendly_error"):
                print(f"❌ Error: {result['user_friendly_error']}")
        else:
            print(f"✅ Web Search Query: {query}")
            response_content = result.get('response_content', 'No response content')
            # Truncate long responses for readability
            if len(response_content) > 300:
                response_content = response_content[:300] + "..."
            print(f"✅ Response: {response_content}")
            
    except Exception as e:
        logger.error(f"Exception in web search test: {e}")
        print(f"❌ Exception: {str(e)}")


async def test_academic_search_capability(client: LangGraphA2AClient) -> None:
    """Test academic paper search capability."""
    logger.info("=" * 60)
    logger.info("TEST 4: Academic search capability")
    logger.info("=" * 60)
    
    query = "Find recent papers on transformer architectures in natural language processing"
    
    try:
        result = await client.send_query(query)
        
        if result.get("error_message"):
            logger.error(f"Error in academic search test: {result['error_message']}")
            if result.get("user_friendly_error"):
                print(f"❌ Error: {result['user_friendly_error']}")
        else:
            print(f"✅ Academic Search Query: {query}")
            response_content = result.get('response_content', 'No response content')
            # Truncate long responses for readability
            if len(response_content) > 300:
                response_content = response_content[:300] + "..."
            print(f"✅ Response: {response_content}")
            
    except Exception as e:
        logger.error(f"Exception in academic search test: {e}")
        print(f"❌ Exception: {str(e)}")


async def test_rag_capability(client: LangGraphA2AClient) -> None:
    """Test RAG document retrieval capability."""
    logger.info("=" * 60)
    logger.info("TEST 5: RAG document retrieval capability")
    logger.info("=" * 60)
    
    query = "What information do the documents contain about policies or requirements?"
    
    try:
        result = await client.send_query(query)
        
        if result.get("error_message"):
            logger.error(f"Error in RAG test: {result['error_message']}")
            if result.get("user_friendly_error"):
                print(f"❌ Error: {result['user_friendly_error']}")
        else:
            print(f"✅ RAG Query: {query}")
            response_content = result.get('response_content', 'No response content')
            # Truncate long responses for readability
            if len(response_content) > 300:
                response_content = response_content[:300] + "..."
            print(f"✅ Response: {response_content}")
            
    except Exception as e:
        logger.error(f"Exception in RAG test: {e}")
        print(f"❌ Exception: {str(e)}")


async def test_error_handling(client: LangGraphA2AClient) -> None:
    """Test error handling capabilities."""
    logger.info("=" * 60)
    logger.info("TEST 6: Error handling")
    logger.info("=" * 60)
    
    # Test with a client configured for wrong URL to trigger connection error
    error_client = LangGraphA2AClient(base_url='http://localhost:9999')
    
    query = "This should fail due to wrong server URL"
    
    try:
        result = await error_client.send_query(query)
        
        if result.get("error_message"):
            print(f"✅ Error handling works: {result.get('user_friendly_error', result['error_message'])}")
        else:
            print("❌ Expected error but got success - error handling may not be working")
            
    except Exception as e:
        print(f"✅ Exception properly caught: {str(e)}")
    finally:
        await error_client.cleanup()


async def run_comprehensive_tests() -> None:
    """Run all test scenarios."""
    print("🚀 Starting LangGraph A2A Client Agent Tests")
    print("=" * 60)
    print("Prerequisites:")
    print("1. A2A server should be running: uv run python -m app")
    print("2. Server should be accessible at http://localhost:10000")
    print("3. .env file should contain OPENAI_API_KEY")
    print("=" * 60)
    
    # Create client
    client = LangGraphA2AClient()
    
    try:
        # Test 1: Single-turn conversation
        task_id, context_id = await test_single_turn_conversation(client)
        
        # Test 2: Multi-turn conversation (if context available)
        if task_id and context_id:
            await test_multi_turn_conversation(client, task_id, context_id)
        
        # Test 3: Web search capability
        await test_web_search_capability(client)
        
        # Test 4: Academic search capability
        await test_academic_search_capability(client)
        
        # Test 5: RAG capability
        await test_rag_capability(client)
        
        # Test 6: Error handling
        await test_error_handling(client)
        
    finally:
        # Clean up
        await client.cleanup()
        print("\n" + "=" * 60)
        print("🏁 Test suite completed!")
        print("=" * 60)


async def interactive_mode():
    """Interactive mode for manual testing."""
    print("🎯 Interactive Mode - LangGraph A2A Client Agent")
    print("=" * 60)
    print("Type your queries below. Type 'quit' to exit, 'help' for guidance.")
    print("=" * 60)
    
    client = LangGraphA2AClient()
    task_id = None
    context_id = None
    
    try:
        while True:
            user_input = input("\n💬 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                print("\n📚 Help:")
                print("- Ask any question to test the A2A client")
                print("- Try: 'What can you help me with?'")
                print("- Try: 'What's the latest news in AI?'")
                print("- Try: 'Find papers on transformers'")
                print("- The client will maintain conversation context automatically")
                continue
            elif not user_input:
                continue
            
            try:
                result = await client.send_query(
                    user_query=user_input,
                    task_id=task_id,
                    context_id=context_id
                )
                
                if result.get("error_message"):
                    print(f"\n❌ Error: {result.get('user_friendly_error', result['error_message'])}")
                else:
                    print(f"\n✅ Response: {result.get('response_content', 'No response content')}")
                    # Update context for next query
                    task_id = result.get('task_id')
                    context_id = result.get('context_id')
                    if task_id:
                        print(f"📝 Context updated (Task: {task_id})")
                        
            except Exception as e:
                print(f"\n❌ Exception: {str(e)}")
                
    finally:
        await client.cleanup()
        print("\n👋 Goodbye!")


async def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        await interactive_mode()
    else:
        await run_comprehensive_tests()


if __name__ == "__main__":
    asyncio.run(main())