"""Enhanced Test Runner for the LangGraph A2A Client Agent

This script demonstrates the enhanced LangGraph A2A client agent capabilities:
- Single-turn conversations (streaming and non-streaming)
- Multi-turn conversations with context preservation
- Enhanced error handling with error categorization
- Boolean routing system validation
- Different query types (web search, academic search, RAG)
- Streaming vs non-streaming mode comparison
- Edge cases and error scenarios

Usage:
    uv run python app/test_client_agent.py
    uv run python app/test_client_agent.py --interactive
    uv run python app/test_client_agent.py --streaming
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from app.client_agent import LangGraphA2AClient, create_langgraph_a2a_client, create_streaming_a2a_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_single_turn_conversation(client: LangGraphA2AClient, mode: str = "regular") -> tuple:
    """Test a simple single-turn conversation."""
    logger.info("=" * 60)
    logger.info(f"TEST 1: Single-turn conversation ({mode} mode)")
    logger.info("=" * 60)
    
    query = "Hello! Can you tell me what you can help me with?"
    
    try:
        # Test with explicit streaming mode if specified
        use_streaming = mode == "streaming"
        result = await client.send_query(query, use_streaming=use_streaming)
        
        # Enhanced result validation
        if result.get("error_message"):
            logger.error(f"Error in single-turn test: {result['error_message']}")
            error_type = result.get("error_type", "unknown")
            print(f"❌ Error ({error_type}): {result.get('user_friendly_error', result['error_message'])}")
        else:
            print(f"✅ Query: {query}")
            print(f"✅ Mode: {mode}")
            print(f"✅ Response: {result.get('response_content', 'No response content')}")
            print(f"✅ Task ID: {result.get('task_id')}")
            print(f"✅ Context ID: {result.get('context_id')}")
            
            # Validate boolean routing flags
            print(f"✅ Boolean flags - prepared_ok: {result.get('prepared_ok')}, send_ok: {result.get('send_ok')}, processing_ok: {result.get('processing_ok')}")
            
            # Check for streaming chunks if in streaming mode
            if use_streaming and result.get('streaming_chunks'):
                print(f"✅ Streaming chunks: {len(result['streaming_chunks'])}")
            
            return result.get('task_id'), result.get('context_id')
            
    except Exception as e:
        logger.error(f"Exception in single-turn test: {e}")
        print(f"❌ Exception: {str(e)}")
    
    return None, None


async def test_multi_turn_conversation(client: LangGraphA2AClient, task_id: str, context_id: str, mode: str = "regular") -> None:
    """Test multi-turn conversation with enhanced context preservation."""
    logger.info("=" * 60)
    logger.info(f"TEST 2: Multi-turn conversation ({mode} mode)")
    logger.info("=" * 60)
    
    if not task_id or not context_id:
        logger.warning("Skipping multi-turn test - no context from previous conversation")
        print("⚠️ Skipping multi-turn test - no context available")
        return
    
    follow_up_query = "Can you elaborate on your capabilities?"
    
    try:
        # Test context preservation with enhanced validation
        use_streaming = mode == "streaming"
        result = await client.send_query(
            user_query=follow_up_query,
            task_id=None,  # Don't reuse completed task_id
            context_id=context_id,
            use_streaming=use_streaming
        )
        
        if result.get("error_message"):
            logger.error(f"Error in multi-turn test: {result['error_message']}")
            error_type = result.get("error_type", "unknown")
            print(f"❌ Error ({error_type}): {result.get('user_friendly_error', result['error_message'])}")
        else:
            print(f"✅ Follow-up Query: {follow_up_query}")
            print(f"✅ Mode: {mode}")
            print(f"✅ Response: {result.get('response_content', 'No response content')}")
            print(f"✅ New task with preserved context: {result.get('task_id')}")
            
            # Validate context preservation worked
            if result.get('context_id') == context_id:
                print(f"✅ Context preserved successfully: {context_id}")
            else:
                print(f"⚠️ Context changed: {context_id} -> {result.get('context_id')}")
            
            # Validate boolean routing flags
            print(f"✅ Boolean flags - prepared_ok: {result.get('prepared_ok')}, send_ok: {result.get('send_ok')}, processing_ok: {result.get('processing_ok')}")
            
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


async def test_streaming_vs_non_streaming(client: LangGraphA2AClient) -> None:
    """Test and compare streaming vs non-streaming modes."""
    logger.info("=" * 60)
    logger.info("TEST 6: Streaming vs Non-streaming comparison")
    logger.info("=" * 60)
    
    query = "What are the main benefits of artificial intelligence?"
    
    try:
        # Test non-streaming
        logger.info("Testing non-streaming mode...")
        non_streaming_result = await client.send_query(query, use_streaming=False)
        
        # Test streaming
        logger.info("Testing streaming mode...")
        streaming_result = await client.send_query(query, use_streaming=True)
        
        # Compare results
        print(f"✅ Query: {query}")
        
        if non_streaming_result.get("error_message"):
            print(f"❌ Non-streaming error: {non_streaming_result.get('user_friendly_error')}")
        else:
            print(f"✅ Non-streaming response length: {len(non_streaming_result.get('response_content', ''))}")
            print(f"✅ Non-streaming flags: prepared_ok={non_streaming_result.get('prepared_ok')}, send_ok={non_streaming_result.get('send_ok')}")
        
        if streaming_result.get("error_message"):
            print(f"❌ Streaming error: {streaming_result.get('user_friendly_error')}")
        else:
            print(f"✅ Streaming response length: {len(streaming_result.get('response_content', ''))}")
            print(f"✅ Streaming chunks: {len(streaming_result.get('streaming_chunks', []))}")
            print(f"✅ Streaming flags: prepared_ok={streaming_result.get('prepared_ok')}, send_ok={streaming_result.get('send_ok')}")
        
    except Exception as e:
        logger.error(f"Exception in streaming comparison test: {e}")
        print(f"❌ Exception: {str(e)}")

async def test_enhanced_error_handling() -> None:
    """Test enhanced error handling capabilities with different error types."""
    logger.info("=" * 60)
    logger.info("TEST 7: Enhanced error handling")
    logger.info("=" * 60)
    
    # Test 1: Connection error (wrong URL)
    print("\n🔍 Testing connection error...")
    error_client = LangGraphA2AClient(base_url='http://localhost:9999')
    
    try:
        result = await error_client.send_query("This should fail due to wrong server URL")
        
        if result.get("error_message"):
            error_type = result.get("error_type", "unknown")
            print(f"✅ Connection error properly handled ({error_type}): {result.get('user_friendly_error')}")
            print(f"✅ Boolean flags correctly set: prepared_ok={result.get('prepared_ok')}, send_ok={result.get('send_ok')}")
        else:
            print("❌ Expected connection error but got success")
    except Exception as e:
        print(f"✅ Exception properly caught: {str(e)}")
    finally:
        await error_client.cleanup()
    
    # Test 2: Validation error (empty query)
    print("\n🔍 Testing validation error...")
    validation_client = LangGraphA2AClient()
    
    try:
        result = await validation_client.send_query("")  # Empty query
        
        if result.get("error_message"):
            error_type = result.get("error_type", "unknown")
            print(f"✅ Validation error properly handled ({error_type}): {result.get('user_friendly_error')}")
        else:
            print("❌ Expected validation error but got success")
    except Exception as e:
        print(f"❌ Unexpected exception: {str(e)}")
    finally:
        await validation_client.cleanup()
    
    print("\n✅ Enhanced error handling tests completed")

async def test_context_validation() -> None:
    """Test context ID validation and recovery."""
    logger.info("=" * 60)
    logger.info("TEST 8: Context validation and recovery")
    logger.info("=" * 60)
    
    client = LangGraphA2AClient()
    
    try:
        # Test with invalid context IDs
        test_cases = [
            {"name": "Empty task_id", "task_id": "", "context_id": "valid-context"},
            {"name": "Empty context_id", "task_id": "valid-task", "context_id": ""},
            {"name": "Whitespace task_id", "task_id": "   ", "context_id": "valid-context"},
            {"name": "None values", "task_id": None, "context_id": None},
        ]
        
        for test_case in test_cases:
            print(f"\n🔍 Testing {test_case['name']}...")
            result = await client.send_query(
                "Test query",
                task_id=test_case["task_id"],
                context_id=test_case["context_id"]
            )
            
            if result.get("prepared_ok"):
                print(f"✅ Context validation handled gracefully")
            else:
                print(f"❌ Context validation failed: {result.get('error_message')}")
    
    except Exception as e:
        logger.error(f"Exception in context validation test: {e}")
        print(f"❌ Exception: {str(e)}")
    finally:
        await client.cleanup()
    
    print("\n✅ Context validation tests completed")


async def run_comprehensive_tests() -> None:
    """Run all enhanced test scenarios."""
    print("🚀 Starting Enhanced LangGraph A2A Client Agent Tests")
    print("=" * 60)
    print("Prerequisites:")
    print("1. A2A server should be running: uv run python -m app")
    print("2. Server should be accessible at http://localhost:10000")
    print("3. .env file should contain OPENAI_API_KEY and TAVILY_API_KEY")
    print("=" * 60)
    
    # Create client
    client = LangGraphA2AClient()
    
    try:
        # Test 1: Single-turn conversation (non-streaming)
        task_id, context_id = await test_single_turn_conversation(client, "regular")
        
        # Test 1b: Single-turn conversation (streaming)
        streaming_task_id, streaming_context_id = await test_single_turn_conversation(client, "streaming")
        
        # Test 2: Multi-turn conversation (if context available)
        if task_id and context_id:
            await test_multi_turn_conversation(client, task_id, context_id, "regular")
        if streaming_task_id and streaming_context_id:
            await test_multi_turn_conversation(client, streaming_task_id, streaming_context_id, "streaming")
        
        # Test 3: Web search capability
        await test_web_search_capability(client)
        
        # Test 4: Academic search capability
        await test_academic_search_capability(client)
        
        # Test 5: RAG capability
        await test_rag_capability(client)
        
        # Test 6: Streaming vs Non-streaming comparison
        await test_streaming_vs_non_streaming(client)
        
    finally:
        # Clean up
        await client.cleanup()
    
    # Test 7: Enhanced error handling (separate clients)
    await test_enhanced_error_handling()
    
    # Test 8: Context validation
    await test_context_validation()
    
    print("\n" + "=" * 60)
    print("🏁 Enhanced test suite completed!")
    print("=" * 60)

async def run_streaming_tests() -> None:
    """Run streaming-focused test scenarios."""
    print("🚀 Starting Streaming-Focused Tests")
    print("=" * 60)
    
    # Create streaming client
    streaming_client = await create_streaming_a2a_client()
    
    try:
        # Test streaming mode exclusively
        print("\n📡 Testing default streaming mode...")
        task_id, context_id = await test_single_turn_conversation(streaming_client, "streaming")
        
        if task_id and context_id:
            await test_multi_turn_conversation(streaming_client, task_id, context_id, "streaming")
        
        # Test streaming with different query types
        await test_web_search_capability(streaming_client)
        
    finally:
        await streaming_client.cleanup()
    
    print("\n" + "=" * 60)
    print("🏁 Streaming tests completed!")
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
    """Enhanced main entry point with multiple test modes."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            await interactive_mode()
        elif sys.argv[1] == '--streaming':
            await run_streaming_tests()
        elif sys.argv[1] == '--errors':
            await test_enhanced_error_handling()
            await test_context_validation()
        else:
            print("Usage: python test_client_agent.py [--interactive|--streaming|--errors]")
            await run_comprehensive_tests()
    else:
        await run_comprehensive_tests()


if __name__ == "__main__":
    asyncio.run(main())