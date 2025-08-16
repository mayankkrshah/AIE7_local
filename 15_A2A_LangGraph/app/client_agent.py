"""LangGraph A2A Client Agent - Activity #1 Implementation

This module implements a LangGraph-based client agent that replicates the behavior
of app/test_client.py, demonstrating agent-to-agent communication through the A2A protocol.

The client agent can:
- Discover AgentCard from A2A server
- Send messages via A2A protocol
- Handle single-turn and multi-turn conversations
- Process streaming and non-streaming responses
- Maintain conversation context with task_id/context_id
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest, SendStreamingMessageRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientAgentState(TypedDict):
    """Enhanced state schema for the LangGraph A2A client agent."""
    # Existing core fields
    messages: Annotated[List, add_messages]
    user_query: str
    a2a_response: Any | None
    task_id: str | None
    context_id: str | None
    
    # Enhanced boolean routing
    prepared_ok: bool | None
    send_ok: bool | None
    processing_ok: bool | None
    
    # Enhanced error handling
    error_message: str | None
    error_type: str | None  # network, parsing, timeout, protocol, validation
    user_friendly_error: str | None
    
    # Streaming support
    use_streaming: bool | None
    streaming_chunks: List[Any] | None
    
    # Enhanced response handling
    response_content: str | None
    response_validated: bool | None
    
    # Internal fields for request preparation
    send_request: Any | None
    
    # Legacy compatibility (deprecated but kept for backward compatibility)
    prepared: bool | None
    send_success: bool | None
    processing_complete: bool | None


class LangGraphA2AClient:
    """LangGraph-based A2A client agent that demonstrates agent-to-agent communication."""
    
    def __init__(self, base_url: str = 'http://localhost:10000'):
        """Initialize the LangGraph A2A client.
        
        Args:
            base_url: Base URL of the A2A server (default: localhost:10000)
        """
        self.base_url = base_url
        self.timeout = httpx.Timeout(60.0)  # Match test_client.py timeout
        self.agent_card: Optional[AgentCard] = None
        self.httpx_client: Optional[httpx.AsyncClient] = None
        self.a2a_client: Optional[A2AClient] = None
        
        # Build the LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph with streaming support and boolean routing."""
        workflow = StateGraph(ClientAgentState)
        
        # Add core nodes
        workflow.add_node("prepare_request", self._prepare_request_node)
        workflow.add_node("route_send_method", self._route_send_method_node)
        workflow.add_node("send_a2a_message", self._send_a2a_message_node)
        workflow.add_node("send_a2a_message_streaming", self._send_a2a_message_streaming_node)
        workflow.add_node("process_response", self._process_response_node)
        workflow.add_node("error_handling", self._error_handling_node)
        
        # Define graph flow
        workflow.set_entry_point("prepare_request")
        
        # Route from prepare_request based on boolean flags
        workflow.add_conditional_edges(
            "prepare_request",
            self._route_after_prepare,
            {
                "success": "route_send_method",
                "error": "error_handling"
            }
        )
        
        # Route between streaming and non-streaming send methods
        workflow.add_conditional_edges(
            "route_send_method",
            self._route_send_method,
            {
                "streaming": "send_a2a_message_streaming",
                "non_streaming": "send_a2a_message"
            }
        )
        
        # Route from both send methods to process or error handling
        workflow.add_conditional_edges(
            "send_a2a_message",
            self._route_after_send,
            {
                "success": "process_response",
                "error": "error_handling"
            }
        )
        
        workflow.add_conditional_edges(
            "send_a2a_message_streaming",
            self._route_after_send,
            {
                "success": "process_response",
                "error": "error_handling"
            }
        )
        
        # Terminal nodes
        workflow.add_edge("process_response", END)
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    async def _prepare_request_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Enhanced request preparation with validation and context handling.
        
        This node:
        - Validates user query (not empty, reasonable length)
        - Validates and sanitizes task_id/context_id
        - Generates unique message_id
        - Creates message payload with user query
        - Sets boolean routing flags
        """
        logger.info(f"Preparing request for query: {state['user_query']}")
        
        try:
            # Enhanced validation
            user_query = state.get('user_query', '').strip()
            if not user_query:
                return self._create_validation_error("User query cannot be empty")
            
            if len(user_query) > 10000:  # Reasonable limit
                return self._create_validation_error("User query too long (max 10,000 characters)")
            
            # Context validation and sanitization
            task_id = self._validate_context_id(state.get('task_id'))
            context_id = self._validate_context_id(state.get('context_id'))
            
            # Generate unique message ID
            message_id = uuid4().hex
            
            # Create message payload
            message_payload = {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': user_query}
                ],
                'message_id': message_id,
            }
            
            # Include conversation context if valid
            if task_id:
                message_payload['task_id'] = task_id
            if context_id:
                message_payload['context_id'] = context_id
            
            # Create SendMessageRequest
            send_request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(message=message_payload)
            )
            
            logger.info(f"Request prepared successfully with message_id: {message_id}")
            
            # Return with enhanced boolean flags
            return {
                "send_request": send_request,
                "prepared_ok": True,
                "prepared": True,  # Legacy compatibility
                "error_message": None,
                "error_type": None,
                "user_friendly_error": None,
                "task_id": task_id,
                "context_id": context_id
            }
            
        except Exception as e:
            logger.error(f"Error preparing request: {e}")
            return self._create_preparation_error(str(e))
    
    def _validate_context_id(self, context_id: str | None) -> str | None:
        """Validate and sanitize context ID."""
        if not context_id:
            return None
        
        context_id = str(context_id).strip()
        if not context_id or context_id.isspace():
            return None
        
        # Basic sanitization - remove potentially problematic characters
        sanitized = ''.join(c for c in context_id if c.isalnum() or c in '-_.')
        return sanitized if sanitized else None
    
    def _create_validation_error(self, message: str) -> Dict[str, Any]:
        """Create a validation error response."""
        return {
            "prepared_ok": False,
            "prepared": False,
            "error_message": message,
            "error_type": "validation",
            "user_friendly_error": message,
            "send_request": None
        }
    
    def _create_preparation_error(self, message: str) -> Dict[str, Any]:
        """Create a preparation error response."""
        return {
            "prepared_ok": False,
            "prepared": False,
            "error_message": f"Error preparing request: {message}",
            "error_type": "preparation",
            "user_friendly_error": "Failed to prepare request. Please try again.",
            "send_request": None
        }
    
    async def _route_send_method_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Route to streaming or non-streaming send method.
        
        This node simply passes through state to allow routing logic
        to determine which send method to use.
        """
        logger.info(f"Routing send method - use_streaming: {state.get('use_streaming')}")
        return {}  # Pass through - routing is handled by _route_send_method
    
    async def _send_a2a_message_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Send message via A2A protocol (non-streaming).
        
        This node:
        - Initializes httpx client and A2A components
        - Resolves AgentCard if not cached
        - Sends message via A2A protocol
        - Enhanced error categorization
        """
        try:
            logger.info("Initializing A2A client and sending message (non-streaming)")
            
            # Initialize components
            await self._initialize_a2a_components()
            
            # Get send request
            send_request = state.get("send_request")
            if not send_request:
                return self._create_network_error("No send_request found in state")
            
            logger.info("Sending message via A2A protocol (non-streaming)")
            response = await self.a2a_client.send_message(send_request)
            
            logger.info("A2A message sent successfully (HTTP 200)")
            
            return {
                "a2a_response": response,
                "send_ok": True,
                "send_success": True,  # Legacy compatibility
                "error_message": None,
                "error_type": None,
                "user_friendly_error": None
            }
            
        except httpx.TimeoutException as e:
            logger.error(f"Timeout sending A2A message: {e}")
            return self._create_timeout_error(str(e))
        except httpx.ConnectError as e:
            logger.error(f"Connection error sending A2A message: {e}")
            return self._create_connection_error(str(e))
        except Exception as e:
            logger.error(f"Error sending A2A message: {e}")
            return self._create_network_error(str(e))
    
    async def _send_a2a_message_streaming_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Send message via A2A protocol (streaming).
        
        This node:
        - Sends streaming message request
        - Aggregates streaming chunks
        - Returns final aggregated response
        """
        try:
            logger.info("Initializing A2A client and sending message (streaming)")
            
            # Initialize components
            await self._initialize_a2a_components()
            
            # Get send request and convert to streaming
            send_request = state.get("send_request")
            if not send_request:
                return self._create_network_error("No send_request found in state")
            
            # Create streaming request
            streaming_request = SendStreamingMessageRequest(
                id=send_request.id,
                params=send_request.params
            )
            
            logger.info("Sending message via A2A protocol (streaming)")
            
            # Collect streaming chunks
            chunks = []
            async for chunk in self.a2a_client.send_message_streaming(streaming_request):
                chunks.append(chunk)
                logger.debug(f"Received streaming chunk: {len(str(chunk))} chars")
            
            # Use the last chunk as the final response (typical streaming behavior)
            final_response = chunks[-1] if chunks else None
            
            if not final_response:
                return self._create_network_error("No streaming response received")
            
            logger.info(f"Streaming message completed with {len(chunks)} chunks")
            
            return {
                "a2a_response": final_response,
                "streaming_chunks": chunks,
                "send_ok": True,
                "send_success": True,  # Legacy compatibility
                "error_message": None,
                "error_type": None,
                "user_friendly_error": None
            }
            
        except httpx.TimeoutException as e:
            logger.error(f"Timeout sending streaming A2A message: {e}")
            return self._create_timeout_error(str(e))
        except httpx.ConnectError as e:
            logger.error(f"Connection error sending streaming A2A message: {e}")
            return self._create_connection_error(str(e))
        except Exception as e:
            logger.error(f"Error sending streaming A2A message: {e}")
            return self._create_network_error(str(e))
    
    async def _initialize_a2a_components(self):
        """Initialize httpx client, agent card, and A2A client if needed."""
        # Initialize httpx client if not exists
        if not self.httpx_client:
            self.httpx_client = httpx.AsyncClient(timeout=self.timeout)
        
        # Resolve AgentCard if not cached
        if not self.agent_card:
            await self._resolve_agent_card()
        
        # Initialize A2A client if not exists
        if not self.a2a_client:
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )
    
    def _create_timeout_error(self, message: str) -> Dict[str, Any]:
        """Create a timeout error response."""
        return {
            "send_ok": False,
            "send_success": False,
            "error_message": f"Timeout: {message}",
            "error_type": "timeout",
            "user_friendly_error": "Request timed out. Please try again or check if the A2A server is running.",
            "a2a_response": None
        }
    
    def _create_connection_error(self, message: str) -> Dict[str, Any]:
        """Create a connection error response."""
        return {
            "send_ok": False,
            "send_success": False,
            "error_message": f"Connection error: {message}",
            "error_type": "network",
            "user_friendly_error": "Cannot connect to A2A server. Please ensure the server is running at http://localhost:10000",
            "a2a_response": None
        }
    
    def _create_network_error(self, message: str) -> Dict[str, Any]:
        """Create a network error response."""
        return {
            "send_ok": False,
            "send_success": False,
            "error_message": f"Network error: {message}",
            "error_type": "network",
            "user_friendly_error": f"Network error occurred: {message}",
            "a2a_response": None
        }
    
    async def _process_response_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Enhanced response processing with defensive parsing.
        
        This node:
        - Validates response structure
        - Extracts response content with multiple fallback strategies
        - Parses task_id and context_id for future messages
        - Validates response content quality
        """
        logger.info("Processing A2A response")
        
        a2a_response = state.get("a2a_response")
        if not a2a_response:
            logger.error("No A2A response to process")
            return self._create_parsing_error("No A2A response to process")
        
        try:
            # Defensive response parsing with multiple strategies
            parsed_result = self._parse_response_with_fallbacks(a2a_response)
            
            if parsed_result["error"]:
                return {
                    "processing_ok": False,
                    "processing_complete": False,
                    "error_message": parsed_result["error"],
                    "error_type": "protocol",
                    "user_friendly_error": "Server returned an error response"
                }
            
            task_id = parsed_result["task_id"]
            context_id = parsed_result["context_id"]
            response_content = parsed_result["content"]
            
            # Validate response content quality
            content_validation = self._validate_response_content(response_content)
            
            logger.info(f"Response processed successfully - task_id: {task_id}, context_id: {context_id}")
            logger.debug(f"Response content: {response_content[:200]}...")  # Log first 200 chars
            
            return {
                "task_id": task_id,
                "context_id": context_id,
                "response_content": response_content,
                "response_validated": content_validation["valid"],
                "processing_ok": True,
                "processing_complete": True,  # Legacy compatibility
                "error_message": None,
                "error_type": None,
                "user_friendly_error": None
            }
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return self._create_parsing_error(str(e))
    
    def _parse_response_with_fallbacks(self, a2a_response) -> Dict[str, Any]:
        """Parse A2A response with multiple fallback strategies."""
        try:
            # Strategy 1: Standard response structure
            if hasattr(a2a_response, 'root'):
                if hasattr(a2a_response.root, 'error'):
                    error_msg = "Unknown server error"
                    if hasattr(a2a_response.root.error, 'message'):
                        error_msg = a2a_response.root.error.message
                    elif hasattr(a2a_response.root.error, 'code'):
                        error_msg = f"Server error code: {a2a_response.root.error.code}"
                    return {"error": error_msg, "task_id": None, "context_id": None, "content": ""}
                
                if hasattr(a2a_response.root, 'result'):
                    result = a2a_response.root.result
                    task_id = getattr(result, 'id', None)
                    context_id = getattr(result, 'context_id', None)
                    
                    # Strategy 1a: Extract from artifacts
                    content = self._extract_content_from_artifacts(result)
                    if content:
                        return {"error": None, "task_id": task_id, "context_id": context_id, "content": content}
                    
                    # Strategy 1b: Extract from message
                    content = self._extract_content_from_message(result)
                    if content:
                        return {"error": None, "task_id": task_id, "context_id": context_id, "content": content}
                    
                    # Strategy 1c: Fallback to string representation
                    content = str(result)
                    return {"error": None, "task_id": task_id, "context_id": context_id, "content": content}
            
            # Strategy 2: Direct response access
            if hasattr(a2a_response, 'content'):
                return {"error": None, "task_id": None, "context_id": None, "content": str(a2a_response.content)}
            
            # Strategy 3: String fallback
            return {"error": None, "task_id": None, "context_id": None, "content": str(a2a_response)}
            
        except Exception as e:
            logger.warning(f"All parsing strategies failed: {e}")
            return {"error": f"Failed to parse response: {str(e)}", "task_id": None, "context_id": None, "content": ""}
    
    def _extract_content_from_artifacts(self, result) -> str:
        """Extract content from response artifacts."""
        content = ""
        try:
            if hasattr(result, 'artifacts') and result.artifacts:
                for artifact in result.artifacts:
                    if hasattr(artifact, 'parts') and artifact.parts:
                        for part in artifact.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                if content:
                                    content += "\n"
                                content += part.root.text
                            elif hasattr(part, 'text'):
                                if content:
                                    content += "\n"
                                content += part.text
        except Exception as e:
            logger.debug(f"Failed to extract from artifacts: {e}")
        return content
    
    def _extract_content_from_message(self, result) -> str:
        """Extract content from response message."""
        content = ""
        try:
            if hasattr(result, 'message') and result.message:
                if hasattr(result.message, 'parts') and result.message.parts:
                    text_parts = []
                    for part in result.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            text_parts.append(part.root.text)
                        elif hasattr(part, 'text'):
                            text_parts.append(part.text)
                    content = '\n'.join(text_parts) if text_parts else str(result.message)
                else:
                    content = str(result.message)
        except Exception as e:
            logger.debug(f"Failed to extract from message: {e}")
        return content
    
    def _validate_response_content(self, content: str) -> Dict[str, Any]:
        """Validate response content quality."""
        if not content or not content.strip():
            return {"valid": False, "reason": "Empty response content"}
        
        if len(content.strip()) < 10:
            return {"valid": False, "reason": "Response too short"}
        
        # Check for obvious error indicators
        content_lower = content.lower()
        error_indicators = ["error:", "exception:", "failed:", "timeout:", "connection refused"]
        for indicator in error_indicators:
            if indicator in content_lower:
                return {"valid": False, "reason": f"Response contains error indicator: {indicator}"}
        
        return {"valid": True, "reason": "Response content validated"}
    
    def _create_parsing_error(self, message: str) -> Dict[str, Any]:
        """Create a parsing error response."""
        return {
            "processing_ok": False,
            "processing_complete": False,
            "error_message": f"Error processing response: {message}",
            "error_type": "parsing",
            "user_friendly_error": "Failed to parse server response. Please try again.",
            "response_content": None,
            "response_validated": False
        }
    
    async def _error_handling_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Enhanced error handling with categorization and recovery suggestions.
        
        This node:
        - Categorizes errors by type
        - Provides user-friendly error messages
        - Suggests recovery actions
        - Maintains error context for debugging
        """
        error_message = state.get("error_message", "Unknown error occurred")
        error_type = state.get("error_type", "unknown")
        existing_user_friendly = state.get("user_friendly_error")
        
        logger.error(f"Handling error (type: {error_type}): {error_message}")
        
        # Generate user-friendly message if not already provided
        if not existing_user_friendly:
            user_friendly_error = self._generate_user_friendly_error(error_message, error_type)
        else:
            user_friendly_error = existing_user_friendly
        
        return {
            "error_message": error_message,
            "error_type": error_type,
            "user_friendly_error": user_friendly_error,
            "error_handled": True,
            # Ensure boolean flags are set to False for errors
            "prepared_ok": False,
            "send_ok": False,
            "processing_ok": False,
            # Legacy compatibility
            "prepared": False,
            "send_success": False,
            "processing_complete": False
        }
    
    def _generate_user_friendly_error(self, error_message: str, error_type: str) -> str:
        """Generate user-friendly error messages based on error type and content."""
        error_message_lower = error_message.lower() if error_message else ""
        
        if error_type == "validation":
            return f"Invalid input: {error_message}"
        elif error_type == "timeout":
            return "Request timed out. Please try again or check if the A2A server is running."
        elif error_type == "network":
            if "connection" in error_message_lower or "refused" in error_message_lower:
                return "Cannot connect to A2A server. Please ensure the server is running at http://localhost:10000"
            elif "timeout" in error_message_lower:
                return "Network timeout occurred. Please check your connection and try again."
            else:
                return f"Network error: {error_message}"
        elif error_type == "parsing":
            return "Failed to parse server response. The server may have returned unexpected data."
        elif error_type == "protocol":
            return "Server returned an error response. Please check your request and try again."
        else:
            return f"An error occurred: {error_message}"
    
    def _route_after_prepare(self, state: ClientAgentState) -> str:
        """Enhanced routing after request preparation using boolean flags."""
        prepared_ok = state.get("prepared_ok", False)
        error_message = state.get("error_message")
        
        logger.info(f"Route after prepare - prepared_ok: {prepared_ok}, error_message: {repr(error_message)}")
        
        # Use enhanced boolean flag for routing
        if prepared_ok and not error_message:
            logger.info("Request preparation successful, proceeding to route send method")
            return "success"
        else:
            logger.error(f"Request preparation failed: {error_message}")
            return "error"
    
    def _route_send_method(self, state: ClientAgentState) -> str:
        """Route between streaming and non-streaming send methods."""
        use_streaming = state.get("use_streaming", False)
        
        logger.info(f"Routing send method - use_streaming: {use_streaming}")
        
        if use_streaming:
            logger.info("Using streaming send method")
            return "streaming"
        else:
            logger.info("Using non-streaming send method")
            return "non_streaming"
    
    def _route_after_send(self, state: ClientAgentState) -> str:
        """Enhanced routing after send based on boolean flags."""
        send_ok = state.get("send_ok", False)
        error_message = state.get("error_message")
        
        logger.info(f"Routing decision - send_ok: {send_ok}, error_message: {repr(error_message)}")
        
        # Use enhanced boolean flag for routing
        if send_ok and not error_message:
            logger.info("Routing to process_response (success path)")
            return "success"
        else:
            logger.info(f"Routing to error_handling (error path) - send_ok: {send_ok}, error: {error_message}")
            return "error"
    
    async def _resolve_agent_card(self):
        """Resolve AgentCard from A2A server."""
        logger.info(f"Resolving AgentCard from {self.base_url}")
        
        resolver = A2ACardResolver(
            httpx_client=self.httpx_client,
            base_url=self.base_url
        )
        
        try:
            # Try to get public agent card
            logger.info(f"Fetching public agent card from: {self.base_url}{AGENT_CARD_WELL_KNOWN_PATH}")
            public_card = await resolver.get_agent_card()
            self.agent_card = public_card
            logger.info("Successfully resolved public agent card")
            
            # Try to get extended card if supported
            if public_card.supports_authenticated_extended_card:
                try:
                    logger.info("Attempting to fetch extended agent card")
                    auth_headers = {'Authorization': 'Bearer dummy-token-for-extended-card'}
                    extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers}
                    )
                    self.agent_card = extended_card
                    logger.info("Successfully resolved extended agent card")
                except Exception as e:
                    logger.warning(f"Failed to fetch extended card, using public card: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to resolve AgentCard: {e}")
            raise
    
    async def send_query(self, user_query: str, task_id: str = None, context_id: str = None, use_streaming: bool = False) -> Dict[str, Any]:
        """Enhanced send query with streaming support and comprehensive state.
        
        Args:
            user_query: The user's query/message
            task_id: Optional task ID for continuing conversation
            context_id: Optional context ID for continuing conversation
            use_streaming: Whether to use streaming mode for the request
            
        Returns:
            Dictionary with response data or error information, including:
            - response_content: The response text
            - task_id/context_id: For conversation continuity
            - Boolean flags: prepared_ok, send_ok, processing_ok
            - Error handling: error_message, error_type, user_friendly_error
            - Streaming: streaming_chunks (if use_streaming=True)
        """
        logger.info(f"Sending query via LangGraph (streaming={use_streaming}): {user_query}")
        
        # Enhanced state initialization with all fields
        initial_state = {
            # Core fields
            "messages": [],
            "user_query": user_query,
            "a2a_response": None,
            "task_id": task_id,
            "context_id": context_id,
            
            # Enhanced boolean routing
            "prepared_ok": None,
            "send_ok": None,
            "processing_ok": None,
            
            # Enhanced error handling
            "error_message": None,
            "error_type": None,
            "user_friendly_error": None,
            
            # Streaming support
            "use_streaming": use_streaming,
            "streaming_chunks": None,
            
            # Enhanced response handling
            "response_content": None,
            "response_validated": None,
            
            # Internal fields
            "send_request": None,
            
            # Legacy compatibility (deprecated but maintained)
            "prepared": None,
            "send_success": None,
            "processing_complete": None
        }
        
        try:
            # Run the enhanced graph
            result = await self.graph.ainvoke(initial_state)
            
            logger.info("LangGraph execution completed")
            logger.debug(f"Final state keys: {list(result.keys())}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during graph execution: {e}")
            # Return comprehensive error state
            return {
                "prepared_ok": False,
                "send_ok": False,
                "processing_ok": False,
                "error_message": f"Graph execution failed: {str(e)}",
                "error_type": "execution",
                "user_friendly_error": "An unexpected error occurred during processing. Please try again.",
                "response_content": None,
                "task_id": None,
                "context_id": None,
                "streaming_chunks": None,
                "response_validated": False
            }
    
    async def cleanup(self):
        """Clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
            logger.info("HTTP client closed")


# Convenience functions for easy usage
async def create_langgraph_a2a_client(base_url: str = 'http://localhost:10000') -> LangGraphA2AClient:
    """Create and return a configured LangGraph A2A client."""
    return LangGraphA2AClient(base_url=base_url)

async def create_streaming_a2a_client(base_url: str = 'http://localhost:10000') -> LangGraphA2AClient:
    """Create and return a LangGraph A2A client configured for streaming."""
    return LangGraphA2AClient(base_url=base_url)