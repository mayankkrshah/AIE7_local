"""
Blockchain Analytics MCP Server
Provides Ethereum blockchain analysis tools via Etherscan API
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
import requests
import json
import certifi
import re

load_dotenv()

# Set up SSL certificates for requests
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Configure requests to use certifi certificates
requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = certifi.where()

# Initialize MCP server
mcp = FastMCP("blockchain-analytics-server")

# Initialize Tavily client for crypto sentiment analysis
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# Etherscan API configuration
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"

def make_etherscan_request(params: dict) -> dict:
    """Helper function to make Etherscan API requests with error handling"""
    try:
        params["apikey"] = ETHERSCAN_API_KEY
        response = session.get(ETHERSCAN_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "1":
            return data
        else:
            return {"error": data.get("message", "Unknown API error")}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}

@mcp.tool()
def get_wallet_balance(address: str) -> str:
    """Get ETH balance and top token balances for any Ethereum address"""
    
    if not address or len(address) != 42 or not address.startswith("0x"):
        return "❌ Invalid Ethereum address format"
    
    try:
        # Get ETH balance
        eth_params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest"
        }
        
        eth_result = make_etherscan_request(eth_params)
        if "error" in eth_result:
            return f"❌ Error fetching ETH balance: {eth_result['error']}"
        
        # Convert Wei to ETH
        eth_balance = int(eth_result["result"]) / 10**18
        
        # Get token balances (top 10)
        token_params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",
            "page": 1,
            "offset": 20
        }
        
        token_result = make_etherscan_request(token_params)
        
        # Process token transactions to find unique tokens
        unique_tokens = {}
        if "error" not in token_result and token_result.get("result"):
            for tx in token_result["result"][:10]:  # Last 10 token transactions
                token_name = tx.get("tokenName", "Unknown")
                token_symbol = tx.get("tokenSymbol", "???")
                if token_symbol not in unique_tokens:
                    unique_tokens[token_symbol] = token_name
        
        # Format results
        result = f"""💰 WALLET ANALYSIS: {address[:8]}...{address[-6:]}

🔹 ETH Balance: {eth_balance:.4f} ETH (${eth_balance * 2000:.2f} USD est.)

📊 Recent Token Activity:
{chr(10).join(f'• {symbol}: {name}' for symbol, name in list(unique_tokens.items())[:5]) if unique_tokens else '• No recent token activity found'}

🔗 Etherscan: https://etherscan.io/address/{address}

📝 Note: Token balances require additional API calls. ETH balance is real-time.
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing wallet: {str(e)}"

@mcp.tool()
def get_transaction_details(tx_hash: str) -> str:
    """Get detailed information about any Ethereum transaction"""
    
    if not tx_hash or len(tx_hash) != 66 or not tx_hash.startswith("0x"):
        return "❌ Invalid transaction hash format"
    
    try:
        # Get transaction details
        tx_params = {
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": tx_hash
        }
        
        tx_result = make_etherscan_request(tx_params)
        if "error" in tx_result:
            return f"❌ Error fetching transaction: {tx_result['error']}"
        
        if not tx_result.get("result"):
            return f"❌ Transaction not found: {tx_hash}"
        
        tx = tx_result["result"]
        
        # Get transaction receipt for status
        receipt_params = {
            "module": "proxy",
            "action": "eth_getTransactionReceipt",
            "txhash": tx_hash
        }
        
        receipt_result = make_etherscan_request(receipt_params)
        receipt = receipt_result.get("result", {})
        
        # Convert values
        value_eth = int(tx.get("value", "0"), 16) / 10**18
        gas_price_gwei = int(tx.get("gasPrice", "0"), 16) / 10**9
        gas_used = int(receipt.get("gasUsed", "0"), 16) if receipt.get("gasUsed") else "Pending"
        
        # Determine status
        status = "✅ Success" if receipt.get("status") == "0x1" else "❌ Failed" if receipt.get("status") == "0x0" else "⏳ Pending"
        
        result = f"""🔍 TRANSACTION ANALYSIS: {tx_hash[:10]}...{tx_hash[-8:]}

{status}

📤 From: {tx.get('from', 'Unknown')[:8]}...{tx.get('from', '')[-6:]}
📥 To: {tx.get('to', 'Contract Creation')[:8]}...{tx.get('to', '')[-6:]}
💰 Value: {value_eth:.6f} ETH
⛽ Gas Price: {gas_price_gwei:.2f} Gwei
🔥 Gas Used: {gas_used}
📦 Block: #{int(tx.get('blockNumber', '0'), 16)}

🔗 Etherscan: https://etherscan.io/tx/{tx_hash}

📝 Input Data: {len(tx.get('input', '0x'))} bytes
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing transaction: {str(e)}"

@mcp.tool()
def analyze_crypto_sentiment(coin: str = "bitcoin", include_price: bool = True) -> str:
    """Analyze market sentiment for any cryptocurrency using news and social signals"""
    
    try:
        # Search for recent news and sentiment
        query = f"{coin} cryptocurrency price prediction sentiment today"
        search_results = client.search(query=query, max_results=10)
        
        # Analyze sentiment from titles and content
        positive_keywords = ['bullish', 'surge', 'rally', 'gains', 'soar', 'breakout', 'moon', 
                            'buy', 'upgrade', 'positive', 'growth', 'rise', 'up', 'high']
        negative_keywords = ['bearish', 'crash', 'dump', 'fall', 'plunge', 'sell', 'warning',
                            'negative', 'down', 'low', 'risk', 'concern', 'fear', 'drop']
        neutral_keywords = ['stable', 'consolidate', 'sideways', 'hold', 'unchanged']
        
        sentiment_scores = []
        key_headlines = []
        price_mentions = []
        
        for result in search_results.get('results', []):
            title = result.get('title', '').lower()
            content = result.get('content', '').lower()[:500]  # First 500 chars
            
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
            
            # Extract price predictions
            price_pattern = r'\$[\d,]+\.?\d*'
            prices = re.findall(price_pattern, result.get('content', ''))
            if prices:
                price_mentions.extend(prices[:2])
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.3:
                overall = "🟢 BULLISH"
                description = "Positive market sentiment detected"
            elif avg_sentiment < -0.3:
                overall = "🔴 BEARISH"
                description = "Negative market sentiment detected"
            else:
                overall = "🟡 NEUTRAL"
                description = "Mixed or neutral market sentiment"
        else:
            overall = "⚪ UNKNOWN"
            description = "Insufficient data for analysis"
        
        # Get current price if requested
        price_info = ""
        if include_price and price_mentions:
            unique_prices = list(set(price_mentions))[:3]
            price_info = f"\n💰 Price Mentions: {', '.join(unique_prices)}"
        
        # Format result
        result = f"""🎯 CRYPTO SENTIMENT ANALYSIS: {coin.upper()}
        
📊 Overall Sentiment: {overall}
📝 {description}

📰 Latest Headlines:
{chr(10).join(key_headlines)}

📈 Sentiment Distribution:
  • Bullish signals: {sum(1 for s in sentiment_scores if s > 0)}
  • Bearish signals: {sum(1 for s in sentiment_scores if s < 0)}
  • Neutral signals: {sum(1 for s in sentiment_scores if s == 0)}
{price_info}

⏰ Analysis based on {len(sentiment_scores)} recent sources
💡 Tip: Combine with on-chain data for better insights

⚠️ Disclaimer: Not financial advice. DYOR."""
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing sentiment for {coin}: {str(e)}"

@mcp.tool()
def track_whale_transactions(min_value_eth: float = 100.0) -> str:
    """Monitor recent large ETH transactions (whales) above specified threshold"""
    
    try:
        # Get latest block
        block_params = {
            "module": "proxy",
            "action": "eth_blockNumber"
        }
        
        block_result = make_etherscan_request(block_params)
        if "error" in block_result:
            return f"❌ Error fetching latest block: {block_result['error']}"
        
        latest_block = int(block_result["result"], 16)
        
        # Get transactions from recent blocks (last 10 blocks)
        whale_txs = []
        for block_offset in range(10):
            block_num = latest_block - block_offset
            
            block_params = {
                "module": "proxy",
                "action": "eth_getBlockByNumber",
                "tag": hex(block_num),
                "boolean": "true"
            }
            
            block_result = make_etherscan_request(block_params)
            if "error" in block_result or not block_result.get("result"):
                continue
            
            block_data = block_result["result"]
            if not block_data.get("transactions"):
                continue
            
            # Check each transaction
            for tx in block_data["transactions"]:
                try:
                    value_eth = int(tx.get("value", "0"), 16) / 10**18
                    if value_eth >= min_value_eth:
                        whale_txs.append({
                            "hash": tx["hash"],
                            "from": tx["from"],
                            "to": tx.get("to", "Contract"),
                            "value": value_eth,
                            "block": block_num
                        })
                except:
                    continue
            
            if len(whale_txs) >= 5:  # Limit to first 5 whale transactions
                break
        
        if not whale_txs:
            return f"🐋 No whale transactions found above {min_value_eth} ETH in recent blocks"
        
        # Format results
        result = f"""🐋 WHALE ALERT - Transactions ≥ {min_value_eth} ETH

Recent large transfers detected:

"""
        
        for i, tx in enumerate(whale_txs[:5], 1):
            result += f"""#{i} 🔸 {tx['value']:.2f} ETH
   📤 From: {tx['from'][:8]}...{tx['from'][-6:]}
   📥 To: {tx['to'][:8]}...{tx['to'][-6:]}
   📦 Block: #{tx['block']}
   🔗 https://etherscan.io/tx/{tx['hash']}

"""
        
        result += f"📊 Found {len(whale_txs)} whale transactions in recent blocks\n"
        result += "⚠️ Note: Real-time whale monitoring requires WebSocket connection"
        
        return result
        
    except Exception as e:
        return f"❌ Error tracking whale transactions: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")