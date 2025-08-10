"""
LangGraph Multi-Domain Intelligence System with Custom State

This application uses the LangGraph StateGraph pattern with custom state
to track analysis progress across multiple domains: crypto, research, finance, and general utilities.
Supports crypto analysis, academic research, financial data, and web search through specialized MCP servers.
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
class MultiDomainAnalysisState(TypedDict):
    """Custom state for multi-domain analysis workflow"""
    messages: Annotated[List[Any], add_messages]  # Required for tool calling
    query: str  # Original user query
    wallet_address: str  # Wallet being analyzed (if any)
    sentiment_data: Dict[str, Any]  # Sentiment analysis results
    balance_data: Dict[str, Any]  # Wallet balance results
    research_data: Dict[str, Any]  # Research results
    financial_data: Dict[str, Any]  # Financial data results
    current_step: str  # Track current step
    final_report: str  # Final analysis report
    intent: str  # Detected query intent (crypto, research, finance, general)
    suggested_tools: List[str]  # Pre-selected relevant tools

class MultiDomainIntelligenceSystem:
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
        
        # Initialize MCP client with all servers
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
            },
            "research": {
                "command": "uv",
                "args": ["--directory", ".", "run", "research_aggregator_server.py"],
                "transport": "stdio",
            }
        })
        
        self.tools = []
        self.graph = None
    
    async def initialize(self):
        """Initialize the system with custom state"""
        try:
            print("🔗 Initializing Multi-Domain Intelligence System...")
            
            # Load all tools from MCP servers
            self.tools = await self.client.get_tools()
            
            print(f"✅ Loaded {len(self.tools)} tools from MCP servers:")
            for tool in self.tools:
                print(f"  • {tool.name}")
            
            # Build the StateGraph with custom state
            self.graph = self._build_custom_state_graph()
            
            print("✅ Multi-Domain StateGraph initialized successfully!")
            return True
                    
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def _build_custom_state_graph(self):
        """Build StateGraph with custom state for multi-domain analysis"""
        print("🔧 Building Multi-Domain StateGraph with custom state...")
        
        # Create workflow with custom state
        workflow = StateGraph(MultiDomainAnalysisState)
        
        # Define nodes
        def prepare_query(state: MultiDomainAnalysisState):
            """Enhanced query preparation with intent detection"""
            import re
            
            query = state.get("query", "").lower()
            
            # Initialize messages if not present
            if not state.get("messages"):
                state["messages"] = [HumanMessage(content=state["query"])]
            
            # Extract wallet address and set intent
            wallet_match = re.search(r'0x[a-fA-F0-9]{40}', state.get("query", ""))
            if wallet_match:
                state["wallet_address"] = wallet_match.group()
                state["intent"] = "wallet_analysis"
                state["suggested_tools"] = ["get_wallet_balance"]
            
            # Detect finance intent (check before crypto to avoid conflicts)
            elif any(keyword in query for keyword in ["stock", "gold", "metal", "silver", "platinum"]):
                state["intent"] = "financial_analysis"
                if any(metal in query for metal in ["gold", "silver", "platinum", "metal"]):
                    state["suggested_tools"] = ["get_metal_price"]
                else:
                    state["suggested_tools"] = ["get_stock_price"]
            
            # Detect crypto sentiment intent (more specific keywords)
            elif any(keyword in query for keyword in ["sentiment", "bitcoin", "ethereum", "crypto", "btc", "eth"]) or \
                 any(crypto_coin in query for crypto_coin in ["bitcoin", "ethereum", "btc", "eth", "solana", "dogecoin"]):
                state["intent"] = "sentiment_analysis"
                # Extract coin name if mentioned
                coin_matches = re.findall(r'\b(bitcoin|ethereum|btc|eth|solana|sol|dogecoin|doge)\b', query)
                if coin_matches:
                    coin = coin_matches[0]
                    # Normalize common abbreviations
                    coin_mapping = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "doge": "dogecoin"}
                    state["target_coin"] = coin_mapping.get(coin, coin)
                state["suggested_tools"] = ["get_crypto_sentiment"]
            
            # Detect research intent
            elif any(keyword in query for keyword in ["search", "papers", "research", "arxiv", "pubmed"]):
                state["intent"] = "research_analysis"
                if "arxiv" in query:
                    state["suggested_tools"] = ["search_arxiv_papers"]
                elif "pubmed" in query:
                    state["suggested_tools"] = ["search_pubmed_papers"]
                else:
                    state["suggested_tools"] = ["aggregate_ai_research"]
            
            # Detect web search intent
            elif any(keyword in query for keyword in ["web", "news", "latest", "current"]):
                state["intent"] = "web_search"
                state["suggested_tools"] = ["web_search"]
            
            # Default to general analysis
            else:
                state["intent"] = "general_analysis"
                state["suggested_tools"] = []
            
            state["current_step"] = "analyzing"
            print(f"🎯 Detected intent: {state.get('intent', 'unknown')}")
            if state.get("suggested_tools"):
                print(f"🔧 Pre-selected tools: {', '.join(state['suggested_tools'])}")
            
            return state
        
        def call_model_with_tools(state: MultiDomainAnalysisState):
            """Call model with smart tool binding based on intent"""
            suggested_tools = state.get("suggested_tools", [])
            
            if suggested_tools:
                # Filter tools to only relevant ones
                relevant_tools = [tool for tool in self.tools if tool.name in suggested_tools]
                print(f"🔥 Using focused tools: {[t.name for t in relevant_tools]}")
                
                if relevant_tools:
                    # For specific intents, call tools directly with extracted parameters
                    if state.get("intent") == "sentiment_analysis" and "get_crypto_sentiment" in suggested_tools:
                        coin = state.get("target_coin", "bitcoin")
                        # Create a focused message that will trigger the sentiment tool
                        focused_message = HumanMessage(content=f"Get crypto sentiment analysis for {coin}")
                        response = self.model.bind_tools(relevant_tools).invoke([focused_message])
                    elif state.get("intent") == "wallet_analysis" and state.get("wallet_address"):
                        # Create a focused message for wallet analysis
                        focused_message = HumanMessage(content=f"Check the ETH balance for wallet {state['wallet_address']}")
                        response = self.model.bind_tools(relevant_tools).invoke([focused_message])
                    else:
                        # Use original message with filtered tools
                        response = self.model.bind_tools(relevant_tools).invoke(state["messages"])
                else:
                    # Fallback to all tools if filtering failed
                    response = self.model.bind_tools(self.tools).invoke(state["messages"])
            else:
                # No specific tools suggested, use all tools
                print("🌍 Using all available tools")
                response = self.model.bind_tools(self.tools).invoke(state["messages"])
            
            return {"messages": [response], "current_step": "tools_called"}
        
        def process_results(state: MultiDomainAnalysisState):
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
        
        # Tools go directly to summarize (no loop back to analyze)
        workflow.add_edge("tools", "summarize")
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
            initial_state: MultiDomainAnalysisState = {
                "messages": [],
                "query": query,
                "wallet_address": "",
                "sentiment_data": {},
                "balance_data": {},
                "research_data": {},
                "financial_data": {},
                "current_step": "starting",
                "final_report": "",
                "intent": "",
                "suggested_tools": []
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
    """Demo of the multi-domain intelligence system"""
    
    print("🚀 Multi-Domain Intelligence System")
    print("=" * 50)
    print("🔧 Supports: Crypto Analysis | Academic Research | Financial Data | Web Search")
    print("=" * 50)
    
    system = MultiDomainIntelligenceSystem()
    
    # Example queries by domain
    example_queries = [
        "🪙 CRYPTO: Check ETH balance for 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "🪙 CRYPTO: What's the current sentiment for bitcoin?",
        "📚 RESEARCH: Search arXiv for papers on large language models",
        "📚 RESEARCH: Find medical AI research on PubMed about cancer diagnosis",
        "📚 RESEARCH: Aggregate AI research from multiple sources",
        "💰 FINANCE: What's the current price of gold?",
        "💰 FINANCE: Get stock price for AAPL",
        "🎲 UTILITY: Roll 2d20 for luck",
        "🌐 WEB: Search web for latest AI news"
    ]
    
    # Get query from command line or use examples
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        queries = [query]
    else:
        print("\nExample queries by domain:")
        for i, q in enumerate(example_queries, 1):
            print(f"{i}. {q}")
        
        choice = input(f"\nEnter query number (1-{len(example_queries)}) or type your own multi-domain query: ").strip()
        
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
        print("📊 MULTI-DOMAIN ANALYSIS RESULT:")
        print("=" * 50)
        print(result)
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())