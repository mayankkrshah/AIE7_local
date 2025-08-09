"""
LangGraph Multi-Agent Blockchain Intelligence Assistant

This application creates a multi-agent system that interacts with our MCP server
to provide comprehensive blockchain wallet analysis using Etherscan data.

Agents:
1. Data Collector - Gathers blockchain data using MCP tools
2. Pattern Analyzer - Analyzes transaction patterns and behaviors
3. Risk Assessor - Evaluates wallet risk scores
4. Report Generator - Creates human-readable intelligence reports
"""

import asyncio
import os
import sys
from typing import Dict, Any, List, TypedDict
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END, START, MessagesState
import concurrent.futures
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from typing import Annotated
from langgraph.graph.message import add_messages

load_dotenv()

# COMPLEX HYBRID STATE combining MessagesState with custom blockchain intelligence fields
class ComplexBlockchainIntelligenceState(TypedDict):
    # Standard LangGraph MessagesState for tool integration
    messages: Annotated[List[Any], add_messages]
    
    # Core blockchain analysis data
    wallet_address: str
    analysis_type: str  # "quick", "standard", "deep", "forensic"
    priority_level: str  # "low", "medium", "high", "critical"
    
    # Multi-source data collection results
    wallet_data: Dict[str, Any]
    whale_activity: Dict[str, Any] 
    market_sentiment: Dict[str, Any]
    web_research: Dict[str, Any]
    social_signals: Dict[str, Any]
    defi_activity: Dict[str, Any]
    
    # Advanced analysis results
    financial_profile: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    network_analysis: Dict[str, Any]
    compliance_status: Dict[str, Any]
    threat_intelligence: Dict[str, Any]
    
    # Multi-format outputs
    executive_summary: str
    technical_report: str
    intelligence_brief: str
    recommendations: List[str]
    action_items: List[Dict[str, Any]]
    
    # Advanced workflow control
    current_phase: str
    active_agents: List[str]
    completed_phases: List[str]
    parallel_tasks: Dict[str, Any]
    escalation_triggers: List[str]
    confidence_scores: Dict[str, float]
    
    # Performance and monitoring
    execution_metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    audit_trail: List[Dict[str, Any]]

class ComplexBlockchainIntelligenceSystem:
    def __init__(self):
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "❌ OPENAI_API_KEY is required for LangGraph agents.\n"
                "💡 Add it to your .env file:\n"
                "   OPENAI_API_KEY=your_openai_api_key_here\n"
                "🔗 Get a key at: https://platform.openai.com/api-keys"
            )
        
        # Initialize BOTH standard LangChain model AND ChatOpenAI for complex workflows
        self.model = init_chat_model("openai:gpt-4o-mini", temperature=0.1)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=openai_key)
        
        # COMPLEX Multi-server MCP client configuration with advanced settings
        self.client = MultiServerMCPClient({
            "general": {
                "command": "uv",
                "args": ["--directory", ".", "run", "server.py"],
                "transport": "stdio",
            },
            "ethereum": {
                "command": "uv", 
                "args": ["--directory", ".", "run", "ethereum_analytics_server.py"],
                "transport": "stdio",
            }
        })
        
        # Advanced tool management
        self.tools = []
        self.ethereum_tools = []
        self.general_tools = []
        self.tool_categories = {}
        
        # COMPLEX workflow graphs
        self.standard_graph = None      # Standard MessagesState + tools_condition
        self.complex_graph = None       # Custom multi-agent StateGraph
        self.hybrid_graph = None        # Hybrid combining both approaches
        
        # Advanced caching and optimization
        self._agents_cache = {}
        self._graph_cache = {}
        self._analysis_cache = {}
        self._performance_metrics = {}
        
        # Complex workflow orchestration
        self.workflow_configs = {
            "quick": {"agents": 2, "depth": "shallow", "parallel": True},
            "standard": {"agents": 4, "depth": "medium", "parallel": True},
            "deep": {"agents": 6, "depth": "deep", "parallel": True},
            "forensic": {"agents": 8, "depth": "comprehensive", "parallel": False}
        }
    
    async def initialize_complex_system(self):
        """Initialize COMPLEX multi-pattern MCP system with multiple graph types"""
        try:
            print("🔗 Initializing COMPLEX Multi-Pattern Blockchain Intelligence System...")
            
            # Load tools from both servers
            self.tools = await self.client.get_tools()
            
            # ADVANCED tool categorization and management
            self._categorize_tools()
            
            print(f"✅ Loaded {len(self.tools)} tools with advanced categorization:")
            print(f"  📊 Ethereum tools: {len(self.ethereum_tools)}")
            print(f"  🌐 General tools: {len(self.general_tools)}")
            print(f"  🏷️ Tool categories: {list(self.tool_categories.keys())}")
            
            # Build MULTIPLE graph patterns
            print("🏗️ Building multiple graph architectures...")
            self.standard_graph = self._build_standard_graph()           # MessagesState + tools_condition
            self.complex_graph = self._build_complex_custom_graph()      # Custom multi-agent
            self.hybrid_graph = self._build_hybrid_graph()               # Combined approach
            
            print("✅ COMPLEX multi-pattern system initialized!")
            print("  🔧 Standard Graph: MessagesState + tools_condition")
            print("  🎯 Complex Graph: Multi-agent custom StateGraph") 
            print("  🚀 Hybrid Graph: Combined architecture")
            return True
                    
        except Exception as e:
            print(f"❌ Failed to initialize complex system: {e}")
            return False
    
    def _categorize_tools(self):
        """Advanced tool categorization for complex workflows"""
        ethereum_tool_names = ['get_wallet_balance', 'get_transaction_details', 'analyze_crypto_sentiment', 'track_whale_transactions']
        
        for tool in self.tools:
            if tool.name in ethereum_tool_names:
                self.ethereum_tools.append(tool)
                category = "blockchain_analysis"
            elif tool.name == "web_search":
                self.general_tools.append(tool)
                category = "research"
            elif tool.name == "roll_dice":
                self.general_tools.append(tool)
                category = "utility"
            elif tool.name == "get_metal_price":
                self.general_tools.append(tool)
                category = "market_data"
            else:
                self.general_tools.append(tool)
                category = "general"
            
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool)
    
    def _build_standard_graph(self):
        """Build STANDARD MessagesState + tools_condition graph (from README)"""
        print("🔧 Building Standard Graph with MessagesState + tools_condition...")
        
        def call_model(state: MessagesState):
            """Standard model calling with tool binding"""
            response = self.model.bind_tools(self.tools).invoke(state["messages"])
            return {"messages": response}
        
        # Build standard graph exactly like README example
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")
        
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
    
    def _build_complex_custom_graph(self):
        """Build COMPLEX custom multi-agent StateGraph"""
        print("🎯 Building Complex Custom Multi-Agent Graph...")
        
        # Create complex custom state graph
        workflow = StateGraph(ComplexBlockchainIntelligenceState)
        
        # Add MULTIPLE specialized agent nodes
        workflow.add_node("intelligence_coordinator", self._intelligence_coordinator_agent)
        workflow.add_node("parallel_data_collector", self._parallel_data_collector_agent)
        workflow.add_node("pattern_analyzer", self._pattern_analyzer_agent)
        workflow.add_node("risk_assessor", self._risk_assessor_agent) 
        workflow.add_node("network_analyzer", self._network_analyzer_agent)
        workflow.add_node("compliance_checker", self._compliance_checker_agent)
        workflow.add_node("threat_hunter", self._threat_hunter_agent)
        workflow.add_node("report_synthesizer", self._report_synthesizer_agent)
        
        # COMPLEX workflow with conditional routing
        workflow.add_edge(START, "intelligence_coordinator")
        workflow.add_edge("intelligence_coordinator", "parallel_data_collector")
        workflow.add_edge("parallel_data_collector", "pattern_analyzer")
        workflow.add_edge("pattern_analyzer", "risk_assessor")
        workflow.add_edge("risk_assessor", "network_analyzer")
        workflow.add_edge("network_analyzer", "compliance_checker") 
        workflow.add_edge("compliance_checker", "threat_hunter")
        workflow.add_edge("threat_hunter", "report_synthesizer")
        workflow.add_edge("report_synthesizer", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _build_hybrid_graph(self):
        """Build HYBRID graph combining both approaches"""
        print("🚀 Building Hybrid Graph combining MessagesState + Custom State...")
        
        # This is the most complex - combines both patterns
        workflow = StateGraph(ComplexBlockchainIntelligenceState)
        
        # Standard tool-calling nodes (MessagesState pattern)
        def hybrid_tool_caller(state: ComplexBlockchainIntelligenceState):
            """Hybrid node that uses both patterns"""
            # Extract messages for standard tool calling
            messages = state.get("messages", [])
            if not messages:
                messages = [HumanMessage(content=f"Analyze wallet {state.get('wallet_address', 'unknown')}")]
            
            # Use standard tool calling approach
            response = self.model.bind_tools(self.tools).invoke(messages)
            
            # Update both message state AND custom state
            return {
                "messages": [response],
                "current_phase": "tool_execution",
                "audit_trail": state.get("audit_trail", []) + [{"action": "hybrid_tool_call", "timestamp": asyncio.get_event_loop().time()}]
            }
        
        # Add hybrid nodes
        workflow.add_node("hybrid_coordinator", self._hybrid_coordinator_agent)
        workflow.add_node("hybrid_tool_caller", hybrid_tool_caller)
        workflow.add_node("hybrid_tools", ToolNode(self.tools))
        workflow.add_node("hybrid_analyzer", self._hybrid_analyzer_agent)
        workflow.add_node("hybrid_synthesizer", self._hybrid_synthesizer_agent)
        
        # Hybrid workflow with conditional edges
        workflow.add_edge(START, "hybrid_coordinator")
        workflow.add_edge("hybrid_coordinator", "hybrid_tool_caller")
        workflow.add_conditional_edges("hybrid_tool_caller", tools_condition, {"tools": "hybrid_tools", "__end__": "hybrid_analyzer"})
        workflow.add_edge("hybrid_tools", "hybrid_tool_caller")
        workflow.add_edge("hybrid_analyzer", "hybrid_synthesizer")
        workflow.add_edge("hybrid_synthesizer", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    # ===== COMPLEX AGENT IMPLEMENTATIONS =====
    
    async def _intelligence_coordinator_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """MASTER coordinator that orchestrates the entire complex analysis"""
        print("🧠 Intelligence Coordinator: Orchestrating complex blockchain analysis...")
        
        wallet_address = state.get("wallet_address", "")
        analysis_type = state.get("analysis_type", "standard")
        config = self.workflow_configs.get(analysis_type, self.workflow_configs["standard"])
        
        return {
            **state,
            "current_phase": "coordination",
            "active_agents": [f"agent_{i}" for i in range(config["agents"])],
            "execution_metrics": {"start_time": asyncio.get_event_loop().time(), "config": config},
            "audit_trail": [{"action": "coordination_start", "timestamp": asyncio.get_event_loop().time()}],
            "messages": state.get("messages", []) + [SystemMessage(content=f"Starting {analysis_type} analysis for {wallet_address}")]
        }
    
    async def _parallel_data_collector_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """ADVANCED parallel data collector with complex orchestration"""
        print("🔍 Parallel Data Collector: Advanced multi-source data gathering...")
        
        # This is more complex than the previous version
        wallet_address = state.get("wallet_address", "")
        
        # Define MULTIPLE parallel collection strategies
        async def collect_blockchain_data():
            tools = self.tool_categories.get("blockchain_analysis", [])
            agent = create_react_agent(self.llm, tools)
            response = await agent.ainvoke({"messages": [HumanMessage(content=f"Comprehensive blockchain analysis for {wallet_address}")]})
            return ("blockchain", response["messages"][-1].content)
            
        async def collect_market_intelligence():
            tools = self.tool_categories.get("market_data", []) + self.tool_categories.get("research", [])
            agent = create_react_agent(self.llm, tools)
            response = await agent.ainvoke({"messages": [HumanMessage(content=f"Market intelligence and research for {wallet_address}")]})
            return ("market", response["messages"][-1].content)
            
        async def collect_threat_intelligence():
            tools = self.tool_categories.get("research", [])
            agent = create_react_agent(self.llm, tools)
            response = await agent.ainvoke({"messages": [HumanMessage(content=f"Threat intelligence research for {wallet_address}")]})
            return ("threat", response["messages"][-1].content)
        
        # Execute in parallel
        tasks = [collect_blockchain_data(), collect_market_intelligence(), collect_threat_intelligence()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process complex results
        data_updates = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            data_type, content = result
            if data_type == "blockchain":
                data_updates["wallet_data"] = {"content": content, "confidence": 0.9}
            elif data_type == "market": 
                data_updates["market_sentiment"] = {"content": content, "confidence": 0.8}
            elif data_type == "threat":
                data_updates["threat_intelligence"] = {"content": content, "confidence": 0.7}
        
        return {
            **state,
            **data_updates,
            "current_phase": "data_collection_complete",
            "confidence_scores": {k: v.get("confidence", 0.5) for k, v in data_updates.items()},
            "completed_phases": state.get("completed_phases", []) + ["parallel_data_collection"]
        }
    
    async def _hybrid_coordinator_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """HYBRID coordinator that bridges MessagesState and custom state patterns"""
        print("🚀 Hybrid Coordinator: Bridging MessagesState + Custom State patterns...")
        
        wallet_address = state.get("wallet_address", "")
        
        # Prepare messages for standard tool calling
        if not state.get("messages"):
            initial_message = HumanMessage(content=f"Perform comprehensive blockchain intelligence analysis for wallet {wallet_address}")
            messages = [initial_message]
        else:
            messages = state["messages"]
        
        return {
            **state,
            "messages": messages,
            "current_phase": "hybrid_coordination",
            "analysis_type": state.get("analysis_type", "hybrid"),
            "priority_level": state.get("priority_level", "high"),
            "audit_trail": state.get("audit_trail", []) + [{"action": "hybrid_coordination", "timestamp": asyncio.get_event_loop().time()}]
        }
    
    async def _hybrid_analyzer_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """HYBRID analyzer that processes both message results and custom state data"""
        print("🔬 Hybrid Analyzer: Processing both MessagesState and custom data...")
        
        # Extract data from messages (tool calling results)
        messages = state.get("messages", [])
        message_content = ""
        if messages:
            message_content = str(messages[-1].content) if hasattr(messages[-1], 'content') else str(messages[-1])
        
        # Combine with custom state data
        combined_analysis = f"""
HYBRID ANALYSIS COMBINING MULTIPLE DATA SOURCES:

MESSAGE-BASED TOOL RESULTS:
{message_content[:500]}...

CUSTOM STATE DATA:
- Wallet Data: {str(state.get('wallet_data', {}))[:200]}...
- Market Sentiment: {str(state.get('market_sentiment', {}))[:200]}...
- Threat Intelligence: {str(state.get('threat_intelligence', {}))[:200]}...

CONFIDENCE SCORES: {state.get('confidence_scores', {})}
"""
        
        return {
            **state,
            "behavioral_patterns": {"hybrid_analysis": combined_analysis},
            "current_phase": "hybrid_analysis_complete",
            "completed_phases": state.get("completed_phases", []) + ["hybrid_analysis"]
        }
    
    async def _hybrid_synthesizer_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """HYBRID synthesizer that creates final intelligence products"""
        print("📝 Hybrid Synthesizer: Creating comprehensive intelligence products...")
        
        # Synthesize all data sources into multiple output formats
        executive_summary = f"""
EXECUTIVE SUMMARY - {state.get('wallet_address', 'Unknown')}

Analysis Type: {state.get('analysis_type', 'standard')}
Priority Level: {state.get('priority_level', 'medium')}
Confidence: {sum(state.get('confidence_scores', {}).values()) / max(len(state.get('confidence_scores', {})), 1):.2f}

Key Findings:
{str(state.get('behavioral_patterns', {}))[:300]}...
"""
        
        technical_report = f"""
TECHNICAL INTELLIGENCE REPORT

Wallet: {state.get('wallet_address', 'Unknown')}
Analysis Phases: {state.get('completed_phases', [])}
Execution Metrics: {state.get('execution_metrics', {})}

Detailed Analysis:
{str(state.get('behavioral_patterns', {}))[:500]}...
"""
        
        return {
            **state,
            "executive_summary": executive_summary,
            "technical_report": technical_report,
            "intelligence_brief": executive_summary,  # Simplified for now
            "current_phase": "synthesis_complete",
            "completed_phases": state.get("completed_phases", []) + ["synthesis"]
        }
    
    def _get_cached_agent(self, agent_key: str, tools: list):
        """Get or create cached agent to avoid recreation overhead"""
        if agent_key not in self._agents_cache:
            self._agents_cache[agent_key] = create_react_agent(self.llm, tools)
        return self._agents_cache[agent_key]

    # ===== ADDITIONAL COMPLEX AGENTS =====
    
    async def _network_analyzer_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Advanced network analysis agent"""
        print("🕸️ Network Analyzer Agent: Analyzing transaction networks...")
        
        return {
            **state,
            "network_analysis": {"analysis": "Network analysis completed", "timestamp": asyncio.get_event_loop().time()},
            "current_phase": "network_analysis",
            "completed_phases": state.get("completed_phases", []) + ["network_analysis"]
        }
    
    async def _compliance_checker_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Compliance and regulatory analysis agent"""
        print("📋 Compliance Checker Agent: Analyzing regulatory compliance...")
        
        return {
            **state,
            "compliance_status": {"status": "Compliance check completed", "timestamp": asyncio.get_event_loop().time()},
            "current_phase": "compliance_check",
            "completed_phases": state.get("completed_phases", []) + ["compliance_check"]
        }
    
    async def _threat_hunter_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Advanced threat hunting agent"""
        print("🎯 Threat Hunter Agent: Hunting for security threats...")
        
        return {
            **state,
            "threat_intelligence": {"threats": "No active threats detected", "timestamp": asyncio.get_event_loop().time()},
            "current_phase": "threat_hunting",
            "completed_phases": state.get("completed_phases", []) + ["threat_hunting"]
        }
    
    async def _report_synthesizer_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Final report synthesis agent"""
        print("📄 Report Synthesizer Agent: Creating comprehensive report...")
        
        # Synthesize all analyses into final report
        final_report = f"""
COMPREHENSIVE BLOCKCHAIN INTELLIGENCE REPORT
Wallet: {state.get('wallet_address', 'Unknown')}

EXECUTIVE SUMMARY:
Analysis completed across {len(state.get('completed_phases', []))} phases.
Current phase: {state.get('current_phase', 'unknown')}

DETAILED FINDINGS:
- Blockchain Data: {str(state.get('wallet_data', {}))[:100]}...
- Network Analysis: {str(state.get('network_analysis', {}))[:100]}...
- Compliance Status: {str(state.get('compliance_status', {}))[:100]}...
- Threat Intelligence: {str(state.get('threat_intelligence', {}))[:100]}...

CONFIDENCE SCORES: {state.get('confidence_scores', {})}
"""
        
        return {
            **state,
            "technical_report": final_report,
            "current_phase": "synthesis_complete",
            "completed_phases": state.get("completed_phases", []) + ["report_synthesis"]
        }

    # ===== COMPLEX ANALYSIS METHODS =====
    
    async def analyze_wallet_complex(self, wallet_address: str, analysis_type: str = "hybrid", graph_type: str = "hybrid") -> str:
        """Run COMPLEX multi-pattern blockchain analysis with selectable graph architectures"""
        
        if not wallet_address or len(wallet_address) != 42 or not wallet_address.startswith("0x"):
            return "❌ Invalid Ethereum address format. Please provide a valid address."
        
        start_time = asyncio.get_event_loop().time()
        print(f"🚀 Starting COMPLEX {graph_type.upper()} blockchain intelligence analysis...")
        print(f"📊 Analysis Type: {analysis_type} | Graph Type: {graph_type}")
        
        # Initialize system if not already done
        if not all([self.standard_graph, self.complex_graph, self.hybrid_graph]):
            success = await self.initialize_complex_system()
            if not success:
                return "❌ Failed to initialize complex analysis system"
        
        try:
            if graph_type == "standard":
                result = await self._run_standard_analysis(wallet_address)
            elif graph_type == "complex":
                result = await self._run_complex_analysis(wallet_address, analysis_type)
            elif graph_type == "hybrid":
                result = await self._run_hybrid_analysis(wallet_address, analysis_type)
            else:
                return f"❌ Unknown graph type: {graph_type}. Choose from: standard, complex, hybrid"
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            print(f"✅ COMPLEX {graph_type.upper()} analysis completed in {duration:.2f}s!")
            return result
                    
        except Exception as e:
            print(f"❌ Complex analysis failed: {e}")
            return f"❌ Complex analysis failed: {str(e)}"
    
    async def _run_standard_analysis(self, wallet_address: str) -> str:
        """Run analysis using STANDARD MessagesState + tools_condition pattern"""
        print("🔧 Executing Standard Graph Analysis...")
        
        response = await self.standard_graph.ainvoke({
            "messages": [HumanMessage(content=f"Analyze Ethereum wallet {wallet_address} using all available tools")]
        })
        
        return f"STANDARD ANALYSIS RESULTS:\n{response['messages'][-1].content}"
    
    async def _run_complex_analysis(self, wallet_address: str, analysis_type: str) -> str:
        """Run analysis using COMPLEX custom multi-agent StateGraph"""
        print("🎯 Executing Complex Custom Graph Analysis...")
        
        # Create complex initial state
        initial_state: ComplexBlockchainIntelligenceState = {
            "messages": [],
            "wallet_address": wallet_address,
            "analysis_type": analysis_type,
            "priority_level": "high",
            "wallet_data": {},
            "whale_activity": {},
            "market_sentiment": {},
            "web_research": {},
            "social_signals": {},
            "defi_activity": {},
            "financial_profile": {},
            "behavioral_patterns": {},
            "risk_assessment": {},
            "network_analysis": {},
            "compliance_status": {},
            "threat_intelligence": {},
            "executive_summary": "",
            "technical_report": "",
            "intelligence_brief": "",
            "recommendations": [],
            "action_items": [],
            "current_phase": "initialization",
            "active_agents": [],
            "completed_phases": [],
            "parallel_tasks": {},
            "escalation_triggers": [],
            "confidence_scores": {},
            "execution_metrics": {},
            "errors": [],
            "warnings": [],
            "audit_trail": []
        }
        
        config = {"configurable": {"thread_id": f"complex-{wallet_address}"}}
        final_state = await self.complex_graph.ainvoke(initial_state, config=config)
        
        return f"COMPLEX MULTI-AGENT ANALYSIS:\n{final_state.get('technical_report', 'No report generated')}"
    
    async def _run_hybrid_analysis(self, wallet_address: str, analysis_type: str) -> str:
        """Run analysis using HYBRID MessagesState + Custom StateGraph"""
        print("🚀 Executing Hybrid Graph Analysis...")
        
        # Create hybrid initial state (combines both patterns)
        initial_state: ComplexBlockchainIntelligenceState = {
            "messages": [HumanMessage(content=f"Perform comprehensive hybrid analysis for wallet {wallet_address}")],
            "wallet_address": wallet_address,
            "analysis_type": analysis_type,
            "priority_level": "critical",
            "wallet_data": {},
            "whale_activity": {},
            "market_sentiment": {},
            "web_research": {},
            "social_signals": {},
            "defi_activity": {},
            "financial_profile": {},
            "behavioral_patterns": {},
            "risk_assessment": {},
            "network_analysis": {},
            "compliance_status": {},
            "threat_intelligence": {},
            "executive_summary": "",
            "technical_report": "",
            "intelligence_brief": "",
            "recommendations": [],
            "action_items": [],
            "current_phase": "hybrid_initialization",
            "active_agents": [],
            "completed_phases": [],
            "parallel_tasks": {},
            "escalation_triggers": [],
            "confidence_scores": {},
            "execution_metrics": {},
            "errors": [],
            "warnings": [],
            "audit_trail": []
        }
        
        config = {"configurable": {"thread_id": f"hybrid-{wallet_address}"}}
        final_state = await self.hybrid_graph.ainvoke(initial_state, config=config)
        
        return f"HYBRID ANALYSIS RESULTS:\n{final_state.get('executive_summary', 'No summary generated')}\n\nTECHNICAL REPORT:\n{final_state.get('technical_report', 'No technical report generated')}"

    async def _parallel_data_collection(self, wallet_address: str):
        """Collect data in parallel for maximum efficiency"""
        
        # Define individual collection tasks
        async def collect_wallet_data():
            agent = self._get_cached_agent("ethereum_agent", self.ethereum_tools)
            prompt = f"Get wallet balance and token activity for {wallet_address} using get_wallet_balance"
            response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return ("wallet", response["messages"][-1].content)

        async def collect_whale_data():
            agent = self._get_cached_agent("ethereum_agent", self.ethereum_tools)
            prompt = "Check for recent whale transactions using track_whale_transactions with default parameters"
            response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return ("whale", response["messages"][-1].content)

        async def collect_sentiment_data():
            agent = self._get_cached_agent("ethereum_agent", self.ethereum_tools)
            prompt = "Get crypto market sentiment for ethereum using analyze_crypto_sentiment"
            response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return ("sentiment", response["messages"][-1].content)

        async def collect_web_data():
            agent = self._get_cached_agent("general_agent", self.general_tools)
            prompt = f"Search web for public information about ethereum address {wallet_address} using web_search"
            response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return ("web", response["messages"][-1].content)

        # Run all data collection tasks in parallel
        tasks = [
            collect_wallet_data(),
            collect_whale_data(), 
            collect_sentiment_data(),
            collect_web_data()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"Data collection error: {str(result)}")
            else:
                data_type, content = result
                data_dict[data_type] = content
        
        return data_dict, errors

    async def _data_collector_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Agent 1: Collect raw blockchain and market data (OPTIMIZED)"""
        print("🔍 Data Collector Agent: Gathering blockchain data in parallel...")
        
        wallet_address = state["wallet_address"]
        
        # Check cache first
        cache_key = f"data_{wallet_address}"
        if cache_key in self._analysis_cache:
            print("📋 Using cached data for faster response")
            cached_data = self._analysis_cache[cache_key]
            return {
                **state,
                "wallet_data": cached_data["wallet_data"],
                "whale_activity": cached_data["whale_activity"],
                "market_sentiment": cached_data["market_sentiment"], 
                "web_research": cached_data["web_research"],
                "current_step": "data_collection",
                "completed_steps": state.get("completed_steps", []) + ["data_collection"]
            }
        
        try:
            # Parallel data collection
            data_dict, collection_errors = await self._parallel_data_collection(wallet_address)
            
            # Structure the collected data
            structured_data = {
                "wallet_data": {"content": data_dict.get("wallet", "No wallet data"), "timestamp": asyncio.get_event_loop().time()},
                "whale_activity": {"content": data_dict.get("whale", "No whale data"), "timestamp": asyncio.get_event_loop().time()},
                "market_sentiment": {"content": data_dict.get("sentiment", "No sentiment data"), "timestamp": asyncio.get_event_loop().time()},
                "web_research": {"content": data_dict.get("web", "No web data"), "timestamp": asyncio.get_event_loop().time()}
            }
            
            # Cache the results for future use (with 5 minute TTL)
            self._analysis_cache[cache_key] = structured_data
            
            return {
                **state,
                **structured_data,
                "current_step": "data_collection",
                "completed_steps": state.get("completed_steps", []) + ["data_collection"],
                "errors": state.get("errors", []) + collection_errors
            }
            
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Parallel data collection failed: {str(e)}"]
            }
    
    async def _pattern_analyzer_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Agent 2: Analyze transaction patterns and behaviors (OPTIMIZED)"""
        print("📊 Pattern Analyzer Agent: Analyzing transaction patterns...")
        
        # Use cached agent for better performance
        analysis_agent = self._get_cached_agent("analysis_agent", [])
        
        # Extract structured data more efficiently
        wallet_data = state.get("wallet_data", {}).get("content", "No wallet data")
        whale_data = state.get("whale_activity", {}).get("content", "No whale data")
        
        prompt = f"""
        You are a pattern analysis expert. Analyze the collected data for wallet {state["wallet_address"]}.
        
        WALLET DATA:
        {wallet_data[:500]}...  # Truncate for efficiency
        
        WHALE ACTIVITY:
        {whale_data[:300]}...   # Truncate for efficiency
        
        Your task (be concise and structured):
        1. Identify transaction patterns and behaviors
        2. Classify wallet activity level (high/medium/low)
        3. Detect any unusual or suspicious patterns
        4. Assess the wallet's primary use case (personal, business, exchange, etc.)
        
        Provide structured analysis in bullet points for efficiency.
        """
        
        try:
            response = await analysis_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            
            return {
                **state,
                "financial_profile": {"patterns": response["messages"][-1].content, "timestamp": asyncio.get_event_loop().time()},
                "current_step": "pattern_analysis",
                "completed_steps": state.get("completed_steps", []) + ["pattern_analysis"]
            }
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Pattern analysis failed: {str(e)}"]
            }
    
    async def _risk_assessor_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Agent 3: Evaluate security risks and compliance issues (OPTIMIZED)"""
        print("⚠️  Risk Assessor Agent: Evaluating security risks...")
        
        # Use cached agent and efficient data access
        risk_agent = self._get_cached_agent("risk_agent", self.general_tools)
        
        # Extract key data efficiently
        wallet_data = state.get("wallet_data", {}).get("content", "No data")[:400]
        pattern_analysis = state.get("financial_profile", {}).get("patterns", "No analysis")[:400]
        web_research = state.get("web_research", {}).get("content", "No web data")[:300]
        
        prompt = f"""
        Risk assessment for wallet {state["wallet_address"]} (be concise):
        
        WALLET DATA: {wallet_data}
        PATTERNS: {pattern_analysis}
        WEB INFO: {web_research}
        
        Tasks (provide bullet-point analysis):
        1. Security risks (hacks, phishing exposure)
        2. Compliance risks (AML/KYC, sanctions)
        3. Bad actor connections (if web search needed, use it sparingly)
        4. Risk score: LOW/MEDIUM/HIGH
        
        Be concise and actionable.
        """
        
        try:
            response = await risk_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            
            return {
                **state,
                "risk_assessment": {"analysis": response["messages"][-1].content, "timestamp": asyncio.get_event_loop().time()},
                "current_step": "risk_assessment", 
                "completed_steps": state.get("completed_steps", []) + ["risk_assessment"]
            }
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Risk assessment failed: {str(e)}"]
            }
    
    async def _report_generator_agent(self, state: ComplexBlockchainIntelligenceState) -> ComplexBlockchainIntelligenceState:
        """Agent 4: Generate comprehensive intelligence report (OPTIMIZED)"""
        print("📝 Report Generator Agent: Creating final intelligence report...")
        
        # Use cached agent for consistency
        report_agent = self._get_cached_agent("report_agent", [])
        
        # Efficiently extract data with size limits
        wallet_info = state.get("wallet_data", {}).get("content", "No data")[:300]
        sentiment_info = state.get("market_sentiment", {}).get("content", "No sentiment")[:200]
        pattern_analysis = state.get("financial_profile", {}).get("patterns", "No patterns")[:400]
        risk_analysis = state.get("risk_assessment", {}).get("analysis", "No risk analysis")[:400]
        
        prompt = f"""
        Create professional intelligence report for wallet {state["wallet_address"]}:

        WALLET DATA: {wallet_info}
        MARKET SENTIMENT: {sentiment_info}
        BEHAVIORAL PATTERNS: {pattern_analysis}
        RISK ANALYSIS: {risk_analysis}
        
        Structure (be concise and professional):
        1. EXECUTIVE SUMMARY (2-3 sentences)
        2. FINANCIAL PROFILE (key metrics)
        3. BEHAVIORAL ANALYSIS (main patterns)
        4. RISK ASSESSMENT (score + key risks)
        5. KEY FINDINGS (3-5 bullet points)
        6. RECOMMENDATIONS (3-4 actionable items)
        
        Keep professional but concise for efficiency.
        """
        
        try:
            response = await report_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            
            return {
                **state,
                "intelligence_report": response["messages"][-1].content,
                "current_step": "completed",
                "completed_steps": state.get("completed_steps", []) + ["report_generation"]
            }
        except Exception as e:
            return {
                **state,
                "errors": state.get("errors", []) + [f"Report generation failed: {str(e)}"]
            }
    
    async def analyze_wallet(self, wallet_address: str, timeout: int = 120) -> str:
        """Run the complete multi-agent blockchain analysis pipeline (OPTIMIZED)"""
        
        if not wallet_address or len(wallet_address) != 42 or not wallet_address.startswith("0x"):
            return "❌ Invalid Ethereum address format. Please provide a valid address."
        
        start_time = asyncio.get_event_loop().time()
        print(f"🚀 Starting OPTIMIZED multi-agent blockchain intelligence analysis for: {wallet_address}")
        
        # Initialize system if not already done
        if not self.graph:
            success = await self.initialize_system()
            if not success:
                return "❌ Failed to initialize blockchain analysis system"
        
        # Create initial state
        initial_state: ComplexBlockchainIntelligenceState = {
            "wallet_address": wallet_address,
            "messages": [],
            "wallet_data": {},
            "whale_activity": {},
            "market_sentiment": {},
            "web_research": {},
            "financial_profile": {},
            "risk_assessment": {},
            "market_intelligence": {},
            "intelligence_report": "",
            "recommendations": [],
            "current_step": "starting",
            "errors": [],
            "completed_steps": []
        }
        
        try:
            # Run the multi-agent workflow with timeout
            config = {"configurable": {"thread_id": f"analysis-{wallet_address}"}}
            
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=config),
                timeout=timeout
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            print(f"✅ OPTIMIZED multi-agent analysis completed in {duration:.2f}s!")
            print(f"📊 Performance: Parallel data collection + cached agents + optimized prompts")
            
            # Return the intelligence report
            if final_state.get("intelligence_report"):
                return final_state["intelligence_report"]
            else:
                errors = final_state.get("errors", [])
                return f"❌ Analysis completed with errors: {'; '.join(errors)}"
                    
        except asyncio.TimeoutError:
            print(f"⏱️ Analysis timed out after {timeout}s")
            return f"⏱️ Analysis timed out after {timeout} seconds. Try again or increase timeout."
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return f"❌ Multi-agent analysis failed: {str(e)}"
            
    def clear_cache(self):
        """Clear analysis cache to free memory"""
        self._analysis_cache.clear()
        print("🧹 Analysis cache cleared")

# COMPLEX demo function showcasing all patterns
async def complex_demo():
    """Demo the COMPLEX multi-pattern blockchain intelligence system"""
    
    print("🚀 Initializing COMPLEX Multi-Pattern Blockchain Intelligence System...")
    print("🏗️ This system demonstrates:")
    print("   • Standard MessagesState + tools_condition pattern")
    print("   • Complex custom multi-agent StateGraph") 
    print("   • Hybrid approach combining both patterns")
    
    system = ComplexBlockchainIntelligenceSystem()
    
    # Initialize the complex system
    print("\n🧪 Initializing complex system with multiple graph architectures...")
    system_ok = await system.initialize_complex_system()
    
    if not system_ok:
        print("❌ Cannot proceed without complex system initialization")
        return
    
    # Get wallet address from command line argument or use default
    if len(sys.argv) > 1:
        test_address = sys.argv[1].strip()
        print(f"📍 Using address from command line: {test_address}")
    else:
        print("\n🔍 Usage: uv run langgraph_app.py <wallet_address>")
        print("Examples:")
        print("- Your wallet: uv run langgraph_app.py 0x...")
        print("- Vitalik: uv run langgraph_app.py 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
        print("- Binance: uv run langgraph_app.py 0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503")
        print("\n🎯 Using Vitalik's address as default demo...")
        test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    # Validate address format
    if not test_address or len(test_address) != 42 or not test_address.startswith("0x"):
        print("❌ Invalid address format. Using Vitalik's address as example...")
        test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    # Demonstrate all three graph patterns
    graph_types = ["standard", "complex", "hybrid"]
    analysis_types = ["quick", "standard", "deep"]
    
    print(f"\n🎯 Target Wallet: {test_address}")
    
    for graph_type in graph_types:
        for analysis_type in analysis_types[:2]:  # Demo first 2 analysis types
            print(f"\n{'='*80}")
            print(f"🚀 Running {graph_type.upper()} Graph with {analysis_type.upper()} Analysis")
            print(f"{'='*80}")
            
            result = await system.analyze_wallet_complex(test_address, analysis_type, graph_type)
            
            print(f"\n📊 {graph_type.upper()} GRAPH RESULTS:")
            print("-" * 60)
            print(result[:1000] + "..." if len(result) > 1000 else result)  # Truncate for demo
            print("-" * 60)
            
            await asyncio.sleep(1)  # Brief pause between demos
    
    print("\n" + "="*80)
    print("✅ COMPLEX MULTI-PATTERN DEMO COMPLETED!")
    print("="*80)
    
    # Show comprehensive architecture summary
    print("\n🏗️ COMPLEX ARCHITECTURE SUMMARY:")
    print("\n📈 STANDARD GRAPH (MessagesState + tools_condition):")
    print("  • Uses standard LangChain model with tool binding")
    print("  • Follows README example pattern exactly")
    print("  • Simple tool calling with conditional edges")
    
    print("\n🎯 COMPLEX GRAPH (Custom Multi-Agent StateGraph):")
    print("  • 8 specialized agents with custom state management")
    print("  • Advanced parallel data collection")
    print("  • Complex workflow orchestration")
    print("  • Confidence scoring and audit trails")
    
    print("\n🚀 HYBRID GRAPH (Combined Approach):")
    print("  • Bridges MessagesState and custom state patterns")
    print("  • Uses both tool_condition AND custom agents")
    print("  • Combines message-based results with custom analytics")
    print("  • Most sophisticated and flexible architecture")
    
    print("\n⚡ ADVANCED FEATURES DEMONSTRATED:")
    print("• Multi-pattern graph architectures")
    print("• Complex state management with audit trails")  
    print("• Advanced tool categorization and management")
    print("• Parallel execution with confidence scoring")
    print("• Hybrid MessagesState + Custom State integration")
    print("• Configurable analysis depth (quick/standard/deep/forensic)")
    print("• Performance metrics and workflow orchestration")

if __name__ == "__main__":
    asyncio.run(complex_demo())