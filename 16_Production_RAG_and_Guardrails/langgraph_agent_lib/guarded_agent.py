"""Production-safe LangGraph agent with Guardrails integration."""

from typing import Dict, Any, List, Optional
import time
import logging
from collections import defaultdict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain

# Configure logging
logger = logging.getLogger(__name__)


class GuardedAgentState(TypedDict):
    """State schema for agent with guard tracking."""
    messages: Annotated[List[BaseMessage], add_messages]
    guard_violations: List[Dict[str, Any]]
    input_validated: bool
    output_validated: bool
    refinement_count: int
    refinement_reason: Optional[str]
    guard_metrics: Dict[str, float]


class GuardMetricsCollector:
    """Collect metrics for guard activations and performance."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "guard_activations": defaultdict(int),
            "guard_latencies": defaultdict(list),
            "blocked_requests": 0,
            "refined_responses": 0
        }
    
    def log_guard_activation(self, guard_name: str, latency: float, blocked: bool = False):
        """Log guard activation metrics."""
        self.metrics["guard_activations"][guard_name] += 1
        self.metrics["guard_latencies"][guard_name].append(latency)
        if blocked:
            self.metrics["blocked_requests"] += 1
    
    def log_request(self):
        """Log a new request."""
        self.metrics["total_requests"] += 1
    
    def log_refinement(self):
        """Log a response refinement."""
        self.metrics["refined_responses"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if self.metrics["total_requests"] == 0:
            return {"message": "No requests processed yet"}
        
        return {
            "total_requests": self.metrics["total_requests"],
            "blocked_rate": self.metrics["blocked_requests"] / self.metrics["total_requests"],
            "refinement_rate": self.metrics["refined_responses"] / self.metrics["total_requests"],
            "avg_latencies": {
                guard: sum(times) / len(times) 
                for guard, times in self.metrics["guard_latencies"].items()
                if times
            },
            "most_active_guard": max(
                self.metrics["guard_activations"].items(), 
                key=lambda x: x[1]
            )[0] if self.metrics["guard_activations"] else None
        }


def create_guarded_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    guards_config: Optional[Dict[str, Any]] = None
):
    """
    Create a production-safe LangGraph agent with integrated guardrails.
    
    This agent adds safety layers for:
    - Input validation (jailbreak, topic, PII detection)
    - Output validation (content moderation, factuality)
    - Error handling and fallback responses
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        guards_config: Configuration for guards
        
    Returns:
        Compiled LangGraph agent with guardrails
    """
    # Import here to avoid circular dependency
    from .agents import get_default_tools
    
    # Set default guards config
    if guards_config is None:
        guards_config = {
            "jailbreak_detection": True,
            "topic_restriction": True,
            "pii_detection": True,
            "content_moderation": True,
            "factuality_check": True,
            "valid_topics": ["student loans", "financial aid", "education financing"],
            "invalid_topics": ["crypto", "gambling", "investment advice", "politics"],
            "max_refinements": 2
        }
    
    # Use same tool setup as simple agent
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Initialize metrics collector
    metrics_collector = GuardMetricsCollector()
    
    # Try to import guardrails (optional dependency)
    try:
        from guardrails import Guard
        from guardrails.hub import (
            RestrictToTopic,
            DetectJailbreak,
            GuardrailsPII,
            ProfanityFree
        )
        guardrails_available = True
    except ImportError:
        logger.warning("Guardrails not available - running without safety guards")
        guardrails_available = False
    
    # Setup guards if available
    input_guards = []
    output_guards = []
    
    if guardrails_available:
        # Input guards
        if guards_config.get("jailbreak_detection"):
            input_guards.append(("jailbreak", Guard().use(DetectJailbreak())))
        
        if guards_config.get("topic_restriction"):
            input_guards.append(("topic", Guard().use(
                RestrictToTopic(
                    valid_topics=guards_config.get("valid_topics", []),
                    invalid_topics=guards_config.get("invalid_topics", []),
                    disable_classifier=True,
                    disable_llm=False,
                    on_fail="exception"
                )
            )))
        
        if guards_config.get("pii_detection"):
            input_guards.append(("pii", Guard().use(
                GuardrailsPII(
                    entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                    on_fail="fix"
                )
            )))
        
        # Output guards
        if guards_config.get("content_moderation"):
            output_guards.append(("profanity", Guard().use(
                ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
            )))
    
    def validate_input(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate user input with guards."""
        messages = state["messages"]
        if not messages:
            return {
                "input_validated": False,
                "guard_violations": [{"guard": "system", "reason": "No input provided"}]
            }
        
        user_input = messages[-1].content
        violations = []
        guard_metrics = {}
        
        # Log request
        metrics_collector.log_request()
        
        # Run through input guards
        for guard_name, guard in input_guards:
            start_time = time.time()
            try:
                result = guard.validate(user_input)
                latency = time.time() - start_time
                guard_metrics[guard_name] = latency
                metrics_collector.log_guard_activation(guard_name, latency)
                
                if not result.validation_passed:
                    violations.append({
                        "guard": guard_name,
                        "reason": str(result.failed_validations)
                    })
                    metrics_collector.log_guard_activation(guard_name, latency, blocked=True)
            except Exception as e:
                latency = time.time() - start_time
                violations.append({
                    "guard": guard_name,
                    "error": str(e)
                })
                metrics_collector.log_guard_activation(guard_name, latency, blocked=True)
        
        return {
            "input_validated": len(violations) == 0,
            "guard_violations": violations,
            "guard_metrics": guard_metrics
        }
    
    def agent_processing(state: GuardedAgentState) -> Dict[str, Any]:
        """Process with the agent (same as simple agent call_model)."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "refinement_count": state.get("refinement_count", 0)
        }
    
    def validate_output(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate agent output with guards."""
        messages = state["messages"]
        if not messages:
            return {"output_validated": False}
        
        agent_response = messages[-1].content
        violations = []
        refined_response = agent_response
        guard_metrics = state.get("guard_metrics", {})
        
        # Run through output guards
        for guard_name, guard in output_guards:
            start_time = time.time()
            try:
                result = guard.validate(refined_response)
                latency = time.time() - start_time
                guard_metrics[guard_name] = latency
                metrics_collector.log_guard_activation(guard_name, latency)
                
                if hasattr(result, 'validated_output') and result.validated_output:
                    refined_response = result.validated_output
                
                if not result.validation_passed:
                    violations.append({
                        "guard": guard_name,
                        "reason": str(result.failed_validations)
                    })
            except Exception as e:
                latency = time.time() - start_time
                violations.append({
                    "guard": guard_name,
                    "error": str(e)
                })
        
        # Update message if refined
        if refined_response != agent_response:
            messages[-1] = AIMessage(content=refined_response)
            metrics_collector.log_refinement()
        
        return {
            "messages": messages,
            "output_validated": len(violations) == 0,
            "guard_violations": violations,
            "guard_metrics": guard_metrics
        }
    
    def refine_response(state: GuardedAgentState) -> Dict[str, Any]:
        """Refine response if validation failed."""
        messages = state["messages"]
        violations = state.get("guard_violations", [])
        
        refinement_prompt = f"""
        The previous response had validation issues: {violations}
        Please provide a corrected response that addresses these concerns.
        Focus on being helpful while staying within guidelines.
        """
        
        refined_response = model.invoke([HumanMessage(content=refinement_prompt)])
        
        metrics_collector.log_refinement()
        
        return {
            "messages": [refined_response],
            "refinement_count": state.get("refinement_count", 0) + 1,
            "refinement_reason": str(violations)
        }
    
    def create_error_response(error_type: str) -> Dict[str, Any]:
        """Create appropriate error response based on violation type."""
        error_messages = {
            "jailbreak": "I cannot process requests that attempt to bypass safety guidelines.",
            "topic": "I can only help with questions about student loans and financial aid.",
            "pii": "Please avoid sharing personal information like SSN or credit card numbers.",
            "profanity": "Please keep our conversation professional.",
            "default": "I encountered an issue processing your request. Please try rephrasing."
        }
        
        message = error_messages.get(error_type, error_messages["default"])
        return {"messages": [AIMessage(content=message)]}
    
    def security_response(state: GuardedAgentState) -> Dict[str, Any]:
        """Response for security violations."""
        return create_error_response("jailbreak")
    
    def topic_redirect(state: GuardedAgentState) -> Dict[str, Any]:
        """Response for off-topic queries."""
        return create_error_response("topic")
    
    def pii_warning(state: GuardedAgentState) -> Dict[str, Any]:
        """Response for PII detection."""
        return create_error_response("pii")
    
    def fallback_response(state: GuardedAgentState) -> Dict[str, Any]:
        """Fallback safe response."""
        return create_error_response("default")
    
    def route_after_input_validation(state: GuardedAgentState) -> str:
        """Route based on input validation results."""
        if state.get("input_validated", False):
            return "agent"
        
        violations = state.get("guard_violations", [])
        for violation in violations:
            guard_type = violation.get("guard", "").lower()
            if "jailbreak" in guard_type:
                return "security_response"
            elif "topic" in guard_type:
                return "topic_redirect"
            elif "pii" in guard_type:
                return "pii_warning"
        
        return "fallback"
    
    def route_after_agent(state: GuardedAgentState) -> str:
        """Route after agent processing."""
        messages = state["messages"]
        if not messages:
            return "fallback"
        
        last_message = messages[-1]
        
        # Check for tool calls (same as simple agent)
        if getattr(last_message, "tool_calls", None):
            return "tools"
        
        return "validate_output"
    
    def route_after_output_validation(state: GuardedAgentState) -> str:
        """Route based on output validation results."""
        if state.get("output_validated", False):
            return END
        
        max_refinements = guards_config.get("max_refinements", 2)
        if state.get("refinement_count", 0) < max_refinements:
            return "refine"
        
        return "fallback"
    
    # Build the graph
    graph = StateGraph(GuardedAgentState)
    
    # Add nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("agent", agent_processing)
    graph.add_node("tools", tool_node)
    graph.add_node("validate_output", validate_output)
    graph.add_node("refine", refine_response)
    graph.add_node("security_response", security_response)
    graph.add_node("topic_redirect", topic_redirect)
    graph.add_node("pii_warning", pii_warning)
    graph.add_node("fallback", fallback_response)
    
    # Set entry point
    graph.set_entry_point("validate_input")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "validate_input",
        route_after_input_validation,
        {
            "agent": "agent",
            "security_response": "security_response",
            "topic_redirect": "topic_redirect",
            "pii_warning": "pii_warning",
            "fallback": "fallback"
        }
    )
    
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "validate_output": "validate_output"
        }
    )
    
    graph.add_edge("tools", "validate_output")
    
    graph.add_conditional_edges(
        "validate_output",
        route_after_output_validation,
        {
            END: END,
            "refine": "refine",
            "fallback": "fallback"
        }
    )
    
    graph.add_edge("refine", "validate_output")
    graph.add_edge("security_response", END)
    graph.add_edge("topic_redirect", END)
    graph.add_edge("pii_warning", END)
    graph.add_edge("fallback", END)
    
    # Compile and attach metrics
    compiled_graph = graph.compile()
    compiled_graph.metrics_collector = metrics_collector
    
    return compiled_graph