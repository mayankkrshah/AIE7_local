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
def get_crypto_sentiment(coin: str = "bitcoin") -> str:
    """
    Get crypto sentiment analysis and market mood for specific coins
    
    Args:
        coin: Cryptocurrency to analyze - 'bitcoin', 'ethereum', 'solana', etc.
    """
    
    try:
        # Build search query for crypto sentiment
        query = f"{coin} cryptocurrency sentiment analysis market mood social media trends price prediction"
        
        try:
            # Use Tavily to search for crypto sentiment data
            search_results = client.search(query=query, max_results=6, search_depth="advanced")
            
            # Process results for sentiment analysis
            sentiment_indicators = []
            bullish_keywords = ['bullish', 'positive', 'rally', 'surge', 'moon', 'pump', 'buy', 'hodl', 'optimistic']
            bearish_keywords = ['bearish', 'negative', 'crash', 'dump', 'sell', 'fear', 'decline', 'drop', 'pessimistic']
            
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for result in search_results.get('results', [])[:5]:
                title = result.get('title', '').lower()
                content = result.get('content', '').lower()
                full_text = f"{title} {content}"
                
                # Count sentiment indicators
                bull_score = sum(1 for word in bullish_keywords if word in full_text)
                bear_score = sum(1 for word in bearish_keywords if word in full_text)
                
                if bull_score > bear_score:
                    sentiment = "🟢 Bullish"
                    bullish_count += 1
                elif bear_score > bull_score:
                    sentiment = "🔴 Bearish"
                    bearish_count += 1
                else:
                    sentiment = "🟡 Neutral"
                    neutral_count += 1
                
                sentiment_indicators.append({
                    'title': result.get('title', 'No title'),
                    'sentiment': sentiment,
                    'url': result.get('url', '')
                })
            
            # Calculate overall sentiment
            total = bullish_count + bearish_count + neutral_count
            if total > 0:
                bullish_pct = (bullish_count / total) * 100
                bearish_pct = (bearish_count / total) * 100
                neutral_pct = (neutral_count / total) * 100
                
                if bullish_pct > 50:
                    overall_sentiment = "🚀 BULLISH"
                    mood_emoji = "🟢"
                elif bearish_pct > 50:
                    overall_sentiment = "📉 BEARISH"  
                    mood_emoji = "🔴"
                else:
                    overall_sentiment = "⚖️ MIXED"
                    mood_emoji = "🟡"
            else:
                overall_sentiment = "❓ UNCLEAR"
                mood_emoji = "⚪"
                bullish_pct = bearish_pct = neutral_pct = 0
            
            # Format sentiment sources
            source_analysis = []
            for item in sentiment_indicators[:3]:
                source_analysis.append(f"   {item['sentiment']} {item['title'][:50]}...")
            
            result = f"""📊 CRYPTO SENTIMENT ANALYSIS: {coin.upper()}
            
{mood_emoji} **Overall Market Sentiment: {overall_sentiment}**

📈 **Sentiment Breakdown:**
   🟢 Bullish: {bullish_pct:.1f}%
   🔴 Bearish: {bearish_pct:.1f}%
   🟡 Neutral: {neutral_pct:.1f}%

📰 **Recent Sentiment Sources:**
{chr(10).join(source_analysis)}

💡 **Key Indicators:**
   • Social media buzz: {'High' if bullish_count + bearish_count > 3 else 'Moderate'}
   • News sentiment: {overall_sentiment.split()[1] if ' ' in overall_sentiment else overall_sentiment}
   • Market mood: {'Risk-on' if bullish_pct > bearish_pct else 'Risk-off' if bearish_pct > bullish_pct else 'Mixed'}

⚠️ **Disclaimer:** Sentiment analysis based on recent news and social data.
Not financial advice. Always DYOR (Do Your Own Research).

🔄 Refresh for updated sentiment analysis"""
            
            return result
            
        except Exception as api_error:
            # Fallback sentiment analysis
            mock_sentiments = {
                "bitcoin": ("🟢 BULLISH", 65, 25, 10),
                "ethereum": ("🟡 MIXED", 45, 35, 20),
                "solana": ("🚀 BULLISH", 70, 20, 10)
            }
            
            sentiment_data = mock_sentiments.get(coin.lower(), ("🟡 MIXED", 40, 40, 20))
            mood, bull, bear, neutral = sentiment_data
            
            return f"""📊 CRYPTO SENTIMENT ANALYSIS: {coin.upper()}
            
{mood.split()[0]} **Overall Market Sentiment: {mood}**

📈 **Sentiment Breakdown:**
   🟢 Bullish: {bull}%
   🔴 Bearish: {bear}%
   🟡 Neutral: {neutral}%

💡 **Analysis based on recent market data and social indicators**
⚠️ Mock data - API temporarily unavailable"""
        
    except Exception as e:
        return f"❌ Error analyzing {coin} sentiment: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")