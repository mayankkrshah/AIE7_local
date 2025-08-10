"""
Crypto Analytics MCP Server
Provides crypto sentiment analysis and wallet balance checking
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
import requests
import json
import re
import certifi

load_dotenv()

# Configure SSL certificates to avoid TLS errors across environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Use a requests session pinned to certifi bundle
requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = certifi.where()

# Initialize MCP server
mcp = FastMCP("crypto-analytics-server")

# Initialize Tavily client for crypto sentiment analysis
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# Etherscan API configuration
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"

@mcp.tool()
def get_wallet_balance(address: str) -> str:
    """Get ETH balance for any Ethereum address"""
    
    if not address or len(address) != 42 or not address.startswith("0x"):
        return "❌ Invalid Ethereum address format"
    
    try:
        # Get ETH balance
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
            "apikey": ETHERSCAN_API_KEY
        }
        
        response = session.get(ETHERSCAN_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "1":
            return f"❌ Error: {data.get('message', 'Failed to fetch balance')}"
        
        # Convert Wei to ETH
        eth_balance = int(data["result"]) / 10**18
        usd_value = eth_balance * 2500  # Approximate USD value
        
        # Format result
        result = f"""💰 WALLET BALANCE

📍 Address: {address[:8]}...{address[-6:]}
💎 Balance: {eth_balance:.4f} ETH
💵 Value: ${usd_value:,.2f} USD (estimated)

🔗 View on Etherscan: https://etherscan.io/address/{address}"""
        
        return result
        
    except Exception as e:
        return f"❌ Error checking wallet balance: {str(e)}"

@mcp.tool()
def analyze_crypto_sentiment(coin: str = "bitcoin") -> str:
    """Analyze market sentiment for any cryptocurrency"""
    
    try:
        # Search for recent news and sentiment
        query = f"{coin} cryptocurrency price prediction sentiment today"
        search_results = client.search(query=query, max_results=10)
        
        # Analyze sentiment from titles and content
        positive_keywords = ['bullish', 'surge', 'rally', 'gains', 'soar', 'breakout', 'moon', 
                            'buy', 'upgrade', 'positive', 'growth', 'rise', 'up', 'high']
        negative_keywords = ['bearish', 'crash', 'dump', 'fall', 'plunge', 'sell', 'warning',
                            'negative', 'down', 'low', 'risk', 'concern', 'fear', 'drop']
        
        sentiment_scores = []
        key_headlines = []
        price_mentions = []
        
        for result in search_results.get('results', []):
            title = result.get('title', '').lower()
            content = result.get('content', '').lower()[:500]
            
            # Score each article
            pos_score = sum(1 for word in positive_keywords if word in title or word in content)
            neg_score = sum(1 for word in negative_keywords if word in title or word in content)
            
            if pos_score > neg_score:
                sentiment_scores.append(1)
                emoji = "📈"
            elif neg_score > pos_score:
                sentiment_scores.append(-1)
                emoji = "📉"
            else:
                sentiment_scores.append(0)
                emoji = "➡️"
            
            # Store key headlines
            if len(key_headlines) < 3:
                key_headlines.append(f"{emoji} {result.get('title', 'No title')[:80]}...")
            
            # Extract price mentions
            price_pattern = r'\$[\d,]+\.?\d*'
            prices = re.findall(price_pattern, result.get('content', ''))
            if prices:
                price_mentions.extend(prices[:2])
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.3:
                overall = "🟢 BULLISH"
                description = "Positive market sentiment"
            elif avg_sentiment < -0.3:
                overall = "🔴 BEARISH"
                description = "Negative market sentiment"
            else:
                overall = "🟡 NEUTRAL"
                description = "Mixed market sentiment"
        else:
            overall = "⚪ UNKNOWN"
            description = "Insufficient data"
        
        # Format result
        result = f"""🎯 CRYPTO SENTIMENT: {coin.upper()}
        
📊 Overall: {overall} - {description}

📰 Latest Headlines:
{chr(10).join(key_headlines)}

📈 Sentiment Stats:
  • Bullish: {sum(1 for s in sentiment_scores if s > 0)} signals
  • Bearish: {sum(1 for s in sentiment_scores if s < 0)} signals
  • Neutral: {sum(1 for s in sentiment_scores if s == 0)} signals

⚠️ Not financial advice. DYOR."""
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing sentiment: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")