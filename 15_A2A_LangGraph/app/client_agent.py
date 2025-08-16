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
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientAgentState(TypedDict):
    """State schema for the LangGraph A2A client agent."""
    messages: Annotated[List, add_messages]
    user_query: str
    a2a_response: Any | None
    task_id: str | None
    context_id: str | None
    error_message: str | None
    send_request: Any | None
    prepared: bool | None
    send_success: bool | None
    processing_complete: bool | None
    response_content: str | None


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
        """Build the LangGraph with 4 core nodes for A2A communication."""
        workflow = StateGraph(ClientAgentState)
        
        # Add nodes
        workflow.add_node("prepare_request", self._prepare_request_node)
        workflow.add_node("send_a2a_message", self._send_a2a_message_node)
        workflow.add_node("process_response", self._process_response_node)
        workflow.add_node("error_handling", self._error_handling_node)
        
        # Define graph flow
        workflow.set_entry_point("prepare_request")
        
        # Conditional flow from prepare_request to handle preparation errors
        workflow.add_conditional_edges(
            "prepare_request",
            self._route_after_prepare,
            {
                "success": "send_a2a_message",
                "error": "error_handling"
            }
        )
        
        # Conditional flow from send_a2a_message based on success/failure
        workflow.add_conditional_edges(
            "send_a2a_message",
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
        """Prepare the A2A request message.
        
        This node:
        - Generates unique message_id
        - Creates message payload with user query
        - Includes task_id/context_id if continuing conversation
        """
        logger.info(f"Preparing request for query: {state['user_query']}")
        
        try:
            # Generate unique message ID
            message_id = uuid4().hex
            
            # Create message payload
            message_payload = {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': state['user_query']}
                ],
                'message_id': message_id,
            }
            
            # Include conversation context if available
            if state.get('task_id'):
                message_payload['task_id'] = state['task_id']
            if state.get('context_id'):
                message_payload['context_id'] = state['context_id']
            
            # Create SendMessageRequest
            send_request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(message=message_payload)
            )
            
            logger.info(f"Request prepared with message_id: {message_id}")
            
            result = {
                "send_request": send_request,
                "prepared": True,
                "error_message": None  # Clear any previous errors
            }
            logger.info(f"Prepare node returning: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error preparing request: {e}")
            return {
                "error_message": f"Error preparing request: {str(e)}",
                "prepared": False
            }
    
    async def _send_a2a_message_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Send message via A2A protocol.
        
        This node:
        - Initializes httpx client and A2A components
        - Resolves AgentCard if not cached
        - Sends message via A2A protocol
        - Handles both streaming and non-streaming responses
        """
        try:
            logger.info("Initializing A2A client and sending message")
            
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
            
            # Send message via A2A protocol
            send_request = state.get("send_request")
            if not send_request:
                raise ValueError("No send_request found in state")
            
            logger.info("Sending message via A2A protocol")
            response = await self.a2a_client.send_message(send_request)
            
            logger.info("A2A message sent successfully (HTTP 200)")
            
            return {
                "a2a_response": response,
                "send_success": True,
                "error_message": None  # Explicitly set to None for successful requests
            }
            
        except Exception as e:
            logger.error(f"Error sending A2A message: {e}")
            return {
                "a2a_response": None,
                "send_success": False,
                "error_message": str(e)
            }
    
    async def _process_response_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Process the A2A response.
        
        This node:
        - Extracts response content
        - Parses task_id and context_id for future messages
        - Formats final response for user
        """
        logger.info("Processing A2A response")
        
        a2a_response = state.get("a2a_response")
        if not a2a_response:
            logger.error("No A2A response to process")
            return {"error_message": "No A2A response to process"}
        
        try:
            # Check if this is an error response
            if hasattr(a2a_response.root, 'error'):
                logger.error(f"A2A response contains error: {a2a_response.root.error}")
                return {
                    "error_message": f"A2A server error: {a2a_response.root.error.message if hasattr(a2a_response.root.error, 'message') else str(a2a_response.root.error)}",
                    "processing_complete": False
                }
            
            # Extract response details from successful response
            if not hasattr(a2a_response.root, 'result'):
                logger.error("A2A response missing result field")
                return {"error_message": "A2A response missing result field"}
                
            result = a2a_response.root.result
            task_id = result.id
            context_id = result.context_id
            
            # Extract response content - handle different response formats
            response_content = ""
            
            # Check for artifacts (main response content)
            if hasattr(result, 'artifacts') and result.artifacts:
                for artifact in result.artifacts:
                    if hasattr(artifact, 'parts') and artifact.parts:
                        for part in artifact.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                if response_content:
                                    response_content += "\n"
                                response_content += part.root.text
            
            # If no artifacts, try to get message content
            if not response_content and hasattr(result, 'message') and result.message:
                if hasattr(result.message, 'parts') and result.message.parts:
                    # Extract text from parts
                    text_parts = []
                    for part in result.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            text_parts.append(part.root.text)
                        elif hasattr(part, 'text'):
                            text_parts.append(part.text)
                    response_content = '\n'.join(text_parts) if text_parts else str(result.message)
                else:
                    response_content = str(result.message)
            
            # Final fallback
            if not response_content:
                response_content = str(result)
            
            logger.info(f"Response processed successfully - task_id: {task_id}, context_id: {context_id}")
            logger.debug(f"Response content: {response_content[:200]}...")  # Log first 200 chars
            
            return {
                "task_id": task_id,
                "context_id": context_id,
                "response_content": response_content,
                "processing_complete": True,
                "error_message": None  # Explicitly clear any error
            }
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return {
                "error_message": f"Error processing response: {str(e)}",
                "processing_complete": False
            }
    
    async def _error_handling_node(self, state: ClientAgentState) -> Dict[str, Any]:
        """Handle errors that occur during A2A communication.
        
        This node:
        - Logs errors appropriately
        - Returns user-friendly error messages
        - Attempts basic error recovery if possible
        """
        error_message = state.get("error_message")
        if error_message is None:
            error_message = "Unknown error occurred"
        
        logger.error(f"Handling error: {error_message}")
        
        # Provide user-friendly error messages
        error_message_lower = error_message.lower() if error_message else ""
        
        if "timeout" in error_message_lower:
            user_message = "Request timed out. Please try again or check if the A2A server is running."
        elif "connection" in error_message_lower:
            user_message = "Cannot connect to A2A server. Please ensure the server is running at http://localhost:10000"
        elif "refused" in error_message_lower:
            user_message = "Connection refused. Please ensure the A2A server is running at http://localhost:10000"
        else:
            user_message = f"An error occurred: {error_message}"
        
        return {
            "error_message": error_message,
            "user_friendly_error": user_message,
            "error_handled": True
        }
    
    def _route_after_prepare(self, state: ClientAgentState) -> str:
        """Route after request preparation based on success."""
        prepared = state.get("prepared", False)
        error_message = state.get("error_message")
        
        logger.info(f"Route after prepare - prepared: {prepared}, error_message: {repr(error_message)}")
        
        # Route to success only if prepared is True and error_message is None/empty
        if prepared and (error_message is None or error_message == ""):
            logger.info("Request preparation successful, proceeding to send")
            return "success"
        else:
            logger.error(f"Request preparation failed: {error_message}")
            return "error"
    
    def _route_after_send(self, state: ClientAgentState) -> str:
        """Route to either process_response or error_handling based on send success."""
        send_success = state.get("send_success", False)
        error_message = state.get("error_message")
        
        logger.info(f"Routing decision - send_success: {send_success}, error_message: {repr(error_message)}")
        
        # Route to success if send_success is True AND no error_message
        if send_success and (error_message is None or error_message == ""):
            logger.info("Routing to process_response (success path)")
            return "success"
        else:
            logger.info(f"Routing to error_handling (error path) - send_success: {send_success}, error: {error_message}")
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
    
    async def send_query(self, user_query: str, task_id: str = None, context_id: str = None) -> Dict[str, Any]:
        """Send a query to the A2A server using the LangGraph.
        
        Args:
            user_query: The user's query/message
            task_id: Optional task ID for continuing conversation
            context_id: Optional context ID for continuing conversation
            
        Returns:
            Dictionary with response data or error information
        """
        logger.info(f"Sending query via LangGraph: {user_query}")
        
        # Initialize state
        initial_state = {
            "messages": [],
            "user_query": user_query,
            "a2a_response": None,
            "task_id": task_id,
            "context_id": context_id,
            "error_message": None,
            "send_request": None,
            "prepared": None,
            "send_success": None,
            "processing_complete": None,
            "response_content": None
        }
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        logger.info("LangGraph execution completed")
        return result
    
    async def cleanup(self):
        """Clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
            logger.info("HTTP client closed")


# Convenience function for easy usage
async def create_langgraph_a2a_client(base_url: str = 'http://localhost:10000') -> LangGraphA2AClient:
    """Create and return a configured LangGraph A2A client."""
    return LangGraphA2AClient(base_url=base_url)