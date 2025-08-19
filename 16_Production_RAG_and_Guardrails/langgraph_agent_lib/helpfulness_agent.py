"""Helpfulness agent implementation following the simple agent pattern."""

from typing import Dict, Any, List, Optional
import re

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain


def extract_location_context(query: str) -> List[str]:
    """
    Extract location-specific context from a query.
    
    Args:
        query: The user query
        
    Returns:
        List of location-related terms found in the query
    """
    # Common location indicators
    us_states = [
        'california', 'texas', 'florida', 'new york', 'pennsylvania', 'illinois',
        'ohio', 'georgia', 'north carolina', 'michigan', 'new jersey', 'virginia',
        'washington', 'arizona', 'massachusetts', 'tennessee', 'indiana', 'missouri',
        'maryland', 'wisconsin', 'colorado', 'minnesota', 'south carolina', 'alabama',
        'louisiana', 'kentucky', 'oregon', 'oklahoma', 'connecticut', 'utah', 'iowa',
        'nevada', 'arkansas', 'mississippi', 'kansas', 'new mexico', 'nebraska',
        'west virginia', 'idaho', 'hawaii', 'new hampshire', 'maine', 'montana',
        'rhode island', 'delaware', 'south dakota', 'north dakota', 'alaska',
        'vermont', 'wyoming', 'dc', 'washington dc', 'district of columbia'
    ]
    
    # Major cities
    major_cities = [
        'new york city', 'los angeles', 'chicago', 'houston', 'phoenix',
        'philadelphia', 'san antonio', 'san diego', 'dallas', 'san jose',
        'austin', 'jacksonville', 'fort worth', 'columbus', 'indianapolis',
        'charlotte', 'san francisco', 'seattle', 'denver', 'washington',
        'boston', 'el paso', 'detroit', 'nashville', 'portland', 'memphis',
        'oklahoma city', 'las vegas', 'louisville', 'baltimore', 'milwaukee',
        'albuquerque', 'tucson', 'fresno', 'mesa', 'sacramento', 'atlanta',
        'kansas city', 'colorado springs', 'miami', 'raleigh', 'omaha',
        'long beach', 'virginia beach', 'oakland', 'minneapolis', 'tulsa'
    ]
    
    # Countries
    countries = [
        'united states', 'usa', 'canada', 'mexico', 'uk', 'united kingdom',
        'germany', 'france', 'spain', 'italy', 'japan', 'china', 'india',
        'australia', 'brazil', 'russia', 'south korea', 'netherlands',
        'switzerland', 'sweden', 'norway', 'denmark', 'finland', 'belgium'
    ]
    
    query_lower = query.lower()
    found_locations = []
    
    # Check for states
    for state in us_states:
        if state in query_lower:
            found_locations.append(state.title())
    
    # Check for cities
    for city in major_cities:
        if city in query_lower:
            found_locations.append(city.title())
    
    # Check for countries
    for country in countries:
        if country in query_lower:
            found_locations.append(country.upper() if country in ['usa', 'uk'] else country.title())
    
    # Check for generic location terms
    if any(term in query_lower for term in ['my state', 'my city', 'my area', 'my region', 'my country']):
        found_locations.append('user\'s specific location')
    
    return found_locations


class HelpfulnessAgentState(TypedDict):
    """State schema for helpfulness agent with evaluation tracking."""
    messages: Annotated[List[BaseMessage], add_messages]
    is_helpful: Optional[bool]
    refinement_count: int
    evaluation_score: float
    max_refinements: int


def evaluate_helpfulness(
    original_query: str,
    response_content: str,
    eval_model
) -> float:
    """
    Evaluate how helpful a response is on a scale of 0-1.
    
    Args:
        original_query: The original user query
        response_content: The agent's response
        eval_model: The model to use for evaluation
        
    Returns:
        Score between 0.0 and 1.0
    """
    # Check if query contains location-specific context
    location_keywords = extract_location_context(original_query)
    location_criteria = ""
    if location_keywords:
        location_criteria = f"""
    - CRITICAL: Does it address the specific location context ({', '.join(location_keywords)})?
    - Does it provide location-specific information rather than generic/national information?
    """
    
    eval_prompt = f"""
    Evaluate the helpfulness of this response on a scale of 0.0 to 1.0.
    
    Original Query: {original_query}
    Response: {response_content}
    
    Consider:
    - Does it directly answer the question?
    - Is the information accurate and specific?{location_criteria}
    - Is it well-structured and clear?
    - Does it provide actionable information?
    - Is the level of detail appropriate?
    
    If the query asks about a specific location but the response provides mostly generic or non-location-specific information, the score should be significantly reduced.
    
    Respond with only a number between 0.0 and 1.0 (e.g., 0.8)
    """
    
    try:
        eval_response = eval_model.invoke([HumanMessage(content=eval_prompt)])
        score_text = eval_response.content.strip()
        
        # Extract numeric score
        score_match = re.search(r'(\d+\.?\d*)', score_text)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    except Exception:
        pass
    
    return 0.5  # Default middle score on any error


def create_helpfulness_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    evaluation_threshold: float = 0.7,
    max_refinements: int = 1
):
    """
    Create a helpfulness-checking LangGraph agent.
    
    This agent follows the same pattern as the simple agent but adds:
    - Automatic helpfulness evaluation after response
    - Response refinement if below threshold
    - Evaluation metrics tracking
    
    The flow is:
    1. Agent generates response (with or without tools)
    2. After final response, evaluate helpfulness
    3. If not helpful enough and under refinement limit, refine
    4. Otherwise, end
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        evaluation_threshold: Minimum helpfulness score (0-1)
        max_refinements: Maximum number of refinement attempts
        
    Returns:
        Compiled LangGraph agent with helpfulness checking
    """
    # Import here to avoid circular dependency
    from .agents import get_default_tools
    
    # Use same tool setup as simple agent
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get models - main model and evaluation model
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Use same model for evaluation with temperature=0 for consistency
    eval_model = get_openai_model(model_name=model_name, temperature=0)
    
    # Create tool node (same as simple agent)
    tool_node = ToolNode(tools)
    
    def call_model(state: HelpfulnessAgentState) -> Dict[str, Any]:
        """
        Invoke the model with messages.
        Enhanced to handle location-specific queries better.
        """
        messages = state["messages"]
        
        # Check if this is the first response (not a refinement)
        if state.get("refinement_count", 0) == 0 and messages:
            # Check for location context in the original query
            original_query = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break
            
            location_keywords = extract_location_context(original_query)
            if location_keywords:
                # Add a system message with location-specific guidance
                location_system_msg = HumanMessage(
                    content=f"""IMPORTANT: The user is asking about {', '.join(location_keywords)}. 
                    Please provide information SPECIFIC to {', '.join(location_keywords)}, not generic or national-level information.
                    Focus on {', '.join(location_keywords)}-specific regulations, programs, timelines, and resources.
                    If you don't have specific information for {', '.join(location_keywords)}, clearly state that and explain what information you can provide instead."""
                )
                # Insert the guidance before invoking the model
                enhanced_messages = messages + [location_system_msg]
                response = model_with_tools.invoke(enhanced_messages)
            else:
                response = model_with_tools.invoke(messages)
        else:
            response = model_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def should_continue_after_agent(state: HelpfulnessAgentState) -> str:
        """
        After agent response, decide whether to:
        - Go to tools (if tool calls)
        - Go to evaluate (if no tool calls)
        """
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        # Check for tool calls
        if last_message and getattr(last_message, "tool_calls", None):
            return "tools"
        
        # No tool calls, go to evaluation
        return "evaluate"
    
    def should_continue_after_evaluate(state: HelpfulnessAgentState) -> str:
        """
        After evaluation, decide whether to:
        - Refine (if not helpful and under limit)
        - End (if helpful or at limit)
        """
        is_helpful = state.get("is_helpful", False)
        refinement_count = state.get("refinement_count", 0)
        max_refine = state.get("max_refinements", max_refinements)
        
        if not is_helpful and refinement_count < max_refine:
            return "refine"
        
        return END
    
    def evaluate(state: HelpfulnessAgentState) -> Dict[str, Any]:
        """
        Evaluate the helpfulness of the last AI response.
        """
        messages = state["messages"]
        if not messages:
            return {"is_helpful": False, "evaluation_score": 0.0}
        
        # Find the original user query
        original_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break
        
        # Find the last AI message (not tool call)
        latest_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                latest_response = msg.content
                break
        
        if not original_query or not latest_response:
            return {"is_helpful": False, "evaluation_score": 0.0}
        
        # Evaluate helpfulness
        score = evaluate_helpfulness(original_query, latest_response, eval_model)
        is_helpful = score >= evaluation_threshold
        
        return {
            "is_helpful": is_helpful,
            "evaluation_score": score
        }
    
    def refine(state: HelpfulnessAgentState) -> Dict[str, Any]:
        """
        Refine the response if not helpful enough.
        """
        messages = state["messages"]
        
        # Find original query
        original_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break
        
        # Find the last response that was evaluated
        last_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                last_response = msg.content
                break
        
        # Check for location-specific context
        location_keywords = extract_location_context(original_query)
        location_guidance = ""
        if location_keywords:
            location_guidance = f"""
        5. IMPORTANT: Focus on {', '.join(location_keywords)}-specific information
        6. Avoid generic or national-level information unless explicitly comparing
        7. Cite {', '.join(location_keywords)}-specific resources, programs, or regulations
        """
        
        refinement_prompt = f"""
        The previous response scored {state.get('evaluation_score', 0):.2f} for helpfulness.
        
        Original question: {original_query}
        
        Previous response: {last_response}
        
        Please provide a MORE comprehensive and helpful response that:
        1. Directly answers the specific question
        2. Provides concrete, actionable information
        3. Uses clear structure and examples where appropriate
        4. Addresses potential follow-up questions{location_guidance}
        
        Focus on being highly specific and relevant to the exact question asked.
        """
        
        # Generate refined response with tools available
        refined_response = model_with_tools.invoke([HumanMessage(content=refinement_prompt)])
        
        return {
            "messages": [refined_response],
            "refinement_count": state.get("refinement_count", 0) + 1
        }
    
    def should_continue_after_refine(state: HelpfulnessAgentState) -> str:
        """
        After refinement, check if we need tools or go to evaluate.
        """
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        # Check for tool calls in refined response
        if last_message and getattr(last_message, "tool_calls", None):
            return "tools"
        
        # No tool calls, go to evaluation
        return "evaluate"
    
    # Build graph following simple agent structure with evaluation
    graph = StateGraph(HelpfulnessAgentState)
    
    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("evaluate", evaluate)
    graph.add_node("refine", refine)
    
    # Set entry point (same as simple agent)
    graph.set_entry_point("agent")
    
    # After agent, either go to tools or evaluate
    graph.add_conditional_edges(
        "agent",
        should_continue_after_agent,
        {
            "tools": "tools",
            "evaluate": "evaluate"
        }
    )
    
    # After tools, go back to agent (same as simple agent)
    graph.add_edge("tools", "agent")
    
    # After evaluate, either refine or end
    graph.add_conditional_edges(
        "evaluate",
        should_continue_after_evaluate,
        {
            "refine": "refine",
            END: END
        }
    )
    
    # After refine, check if tools needed or evaluate
    graph.add_conditional_edges(
        "refine",
        should_continue_after_refine,
        {
            "tools": "tools",
            "evaluate": "evaluate"
        }
    )
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    # Wrap to ensure proper initial state
    class HelpfulnessAgentWrapper:
        def __init__(self, graph):
            self.graph = graph
            self.max_refinements_default = max_refinements
        
        def invoke(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
            """Ensure all required state fields are initialized."""
            # Set defaults
            default_state = {
                "is_helpful": None,
                "refinement_count": 0,
                "evaluation_score": 0.0,
                "max_refinements": self.max_refinements_default
            }
            
            # Merge with input, preserving input values
            full_state = {**default_state, **input_state}
            
            # Invoke the graph
            return self.graph.invoke(full_state)
        
        def get_graph(self):
            return self.graph.get_graph() if hasattr(self.graph, 'get_graph') else None
    
    # Return the wrapped graph
    return HelpfulnessAgentWrapper(compiled_graph)