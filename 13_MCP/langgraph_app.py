"""
LangGraph Crypto Intelligence System with Custom State

This application uses the LangGraph StateGraph pattern with custom state
to track analysis progress through our MCP tools.
"""

import asyncio
import os
import sys
from typing import Dict, Any, List, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()

# Custom State Definition
class CryptoAnalysisState(TypedDict):
    """Custom state for crypto analysis workflow"""
    messages: Annotated[List[Any], add_messages]  # Required for tool calling
    query: str  # Original user query
    wallet_address: str  # Wallet being analyzed (if any)
    sentiment_data: Dict[str, Any]  # Sentiment analysis results
    balance_data: Dict[str, Any]  # Wallet balance results
    current_step: str  # Track current step
    final_report: str  # Final analysis report

class CryptoIntelligenceSystem:
    def __init__(self):
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "❌ OPENAI_API_KEY is required for LangGraph.\n"
                "💡 Add it to your .env file"
            )
        
        # Initialize model
        self.model = init_chat_model("openai:gpt-4o-mini", temperature=0.1)
        
        # Initialize MCP client with both servers
        self.client = MultiServerMCPClient({
            "main": {
                "command": "uv",
                "args": ["--directory", ".", "run", "server.py"],
                "transport": "stdio",
            },
            "crypto": {
                "command": "uv", 
                "args": ["--directory", ".", "run", "ethereum_analytics_server.py"],
                "transport": "stdio",
            }
        })
        
        self.tools = []
        self.graph = None
    
    async def initialize(self):
        """Initialize the system with custom state"""
        try:
            print("🔗 Initializing Crypto Intelligence System...")
            
            # Load all tools from MCP servers
            self.tools = await self.client.get_tools()
            
            print(f"✅ Loaded {len(self.tools)} tools from MCP servers:")
            for tool in self.tools:
                print(f"  • {tool.name}")
            
            # Build the StateGraph with custom state
            self.graph = self._build_custom_state_graph()
            
            print("✅ Custom StateGraph initialized successfully!")
            return True
                    
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def _build_custom_state_graph(self):
        """Build StateGraph with custom state"""
        print("🔧 Building StateGraph with custom state...")
        
        # Create workflow with custom state
        workflow = StateGraph(CryptoAnalysisState)
        
        # Define nodes
        def prepare_query(state: CryptoAnalysisState):
            """Prepare the initial query"""
            # Initialize messages if not present
            if not state.get("messages"):
                state["messages"] = [HumanMessage(content=state["query"])]
            
            # Extract wallet address if present in query
            import re
            wallet_match = re.search(r'0x[a-fA-F0-9]{40}', state.get("query", ""))
            if wallet_match:
                state["wallet_address"] = wallet_match.group()
            
            state["current_step"] = "analyzing"
            return state
        
        def call_model_with_tools(state: CryptoAnalysisState):
            """Call model with tool binding"""
            response = self.model.bind_tools(self.tools).invoke(state["messages"])
            return {"messages": [response], "current_step": "tools_called"}
        
        def process_results(state: CryptoAnalysisState):
            """Process and summarize results"""
            # Extract data from messages
            messages = state.get("messages", [])
            
            # Create final report
            report_parts = []
            
            if state.get("wallet_address"):
                report_parts.append(f"📍 Wallet Analysis: {state['wallet_address']}")
            
            # Add message content to report
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    report_parts.append(str(msg.content))
            
            state["final_report"] = "\n\n".join(report_parts)
            state["current_step"] = "completed"
            return state
        
        # Add nodes
        workflow.add_node("prepare", prepare_query)
        workflow.add_node("analyze", call_model_with_tools)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("summarize", process_results)
        
        # Add edges
        workflow.add_edge(START, "prepare")
        workflow.add_edge("prepare", "analyze")
        
        # Conditional edge for tool calling
        workflow.add_conditional_edges(
            "analyze",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "summarize"
            }
        )
        
        workflow.add_edge("tools", "analyze")
        workflow.add_edge("summarize", END)
        
        # Add memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def analyze(self, query: str) -> str:
        """Run analysis using custom state"""
        
        print(f"\n🚀 Processing query: {query}")
        
        # Initialize if needed
        if not self.graph:
            success = await self.initialize()
            if not success:
                return "❌ Failed to initialize system"
        
        try:
            # Create initial state
            initial_state: CryptoAnalysisState = {
                "messages": [],
                "query": query,
                "wallet_address": "",
                "sentiment_data": {},
                "balance_data": {},
                "current_step": "starting",
                "final_report": ""
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": f"analysis-{hash(query)}"}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # Return the final report or last message
            if final_state.get("final_report"):
                return final_state["final_report"]
            elif final_state.get("messages"):
                last_msg = final_state["messages"][-1]
                return last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            else:
                return "Analysis completed but no results found."
                    
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"

async def main():
    """Demo of the custom state crypto intelligence system"""
    
    print("🚀 Crypto Intelligence System (Custom State)")
    print("=" * 50)
    
    system = CryptoIntelligenceSystem()
    
    # Example queries
    example_queries = [
        "Check the ETH balance for wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "What's the current sentiment for bitcoin?",
        "Analyze ethereum market sentiment",
        "Roll 2d20 for luck",
        "Search web for latest crypto regulations"
    ]
    
    # Get query from command line or use examples
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        queries = [query]
    else:
        print("\nExample queries available:")
        for i, q in enumerate(example_queries, 1):
            print(f"{i}. {q}")
        
        choice = input("\nEnter query number (1-5) or type your own query: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(example_queries):
            queries = [example_queries[int(choice) - 1]]
        elif choice:
            queries = [choice]
        else:
            print("Using first example query...")
            queries = [example_queries[0]]
    
    # Process the query
    for query in queries:
        result = await system.analyze(query)
        
        print("\n" + "=" * 50)
        print("📊 RESULT:")
        print("=" * 50)
        print(result)
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())