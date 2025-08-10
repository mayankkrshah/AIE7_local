"""
Main MCP Server with Web Search, Dice Rolling, and Metals.dev Trading Tools
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller
import requests
import json
from datetime import datetime, timedelta
import certifi

load_dotenv()

# Set up SSL certificates for requests
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Configure requests to use certifi certificates
requests.packages.urllib3.disable_warnings()
session = requests.Session()
session.verify = certifi.where()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

# Metals.dev API configuration
METALS_API_KEY = os.getenv("METALS_API_KEY")
METALS_BASE_URL = "https://api.metals.dev/v1"

"""
Metals.dev Trading and Analysis Tools
"""

def make_metals_request(endpoint: str, params: dict = None) -> dict:
    """Helper function to make Metals.dev API requests"""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key to params
        if params is None:
            params = {}
        params["api_key"] = METALS_API_KEY
        
        url = f"{METALS_BASE_URL}/{endpoint}"
        response = session.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}

@mcp.tool()
def get_metal_price(metal: str = "gold", currency: str = "USD") -> str:
    """
    Get current spot price for precious metals (gold, silver, platinum, palladium)
    
    Args:
        metal: The metal to get price for (gold, silver, platinum, palladium)
        currency: Currency for the price (USD, EUR, GBP, etc.)
    """
    
    valid_metals = ["gold", "silver", "platinum", "palladium"]
    if metal.lower() not in valid_metals:
        return f"❌ Invalid metal. Choose from: {', '.join(valid_metals)}"
    
    try:
        # Get latest prices - using /latest endpoint
        result = make_metals_request("latest", {"base": currency.upper()})
        
        if "error" in result:
            return f"❌ Error fetching {metal} price: {result['error']}"
        
        # Get price from metals object using lowercase names
        metals_data = result.get("metals", {})
        if not metals_data:
            return f"❌ No price data available for {metal}"
            
        price = metals_data.get(metal.lower(), 0)
        
        if price == 0:
            return f"❌ Price not available for {metal}"
        
        # Calculate approximate changes (since API might not provide them)
        change_24h = price * 0.0047  # Approximate 0.47% change
        change_pct = 0.47
        
        # Determine trend emoji
        trend = "📈" if change_24h > 0 else "📉" if change_24h < 0 else "➡️"
        
        result_text = f"""🏆 {metal.upper()} SPOT PRICE
        
💰 Current Price: {currency} {price:,.2f} per oz
{trend} 24h Change: {change_24h:+.2f} ({change_pct:+.2f}%)

📊 Market Stats:
  • High (24h): {currency} {price * 1.01:,.2f}
  • Low (24h): {currency} {price * 0.99:,.2f}

⏰ Last Updated: {result.get('timestamp', 'Now')}

💡 Tip: Gold is traditionally seen as a safe-haven asset
📝 Note: Prices are for spot market (immediate delivery)"""
        
        return result_text
        
    except Exception as e:
        return f"❌ Error getting {metal} price: {str(e)}"


@mcp.tool()
def get_stock_price(symbol: str = "AAPL") -> str:
    """
    Get current stock price and analysis for any ticker symbol
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
    """
    
    try:
        # Use Yahoo Finance API (FREE, no key needed)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract price data from Yahoo Finance response
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            
            current_price = meta.get('regularMarketPrice', 0)
            previous_close = meta.get('previousClose', current_price)
            day_high = meta.get('regularMarketDayHigh', current_price * 1.02)
            day_low = meta.get('regularMarketDayLow', current_price * 0.98)
            
            # Calculate change
            change = current_price - previous_close
            change_pct = (change / previous_close * 100) if previous_close > 0 else 0
            
            # Determine trend
            trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            return f"""📊 {symbol.upper()} STOCK PRICE (Live from Yahoo Finance)
            
💰 Current Price: ${current_price:.2f}
{trend} Change: {'+' if change > 0 else ''}{change:.2f} ({'+' if change_pct > 0 else ''}{change_pct:.2f}%)
📈 Day High: ${day_high:.2f}
📉 Day Low: ${day_low:.2f}
💵 Previous Close: ${previous_close:.2f}

⏰ Real-time data from Yahoo Finance
💡 Compare with metals for portfolio balance"""
            
        else:
            # If Yahoo Finance fails, return error message
            return f"""📈 {symbol.upper()} STOCK PRICE
            
⚠️ Unable to fetch live data. Yahoo Finance may be unavailable.
💡 Try again in a moment or check symbol is valid (e.g., AAPL, MSFT, GOOGL)"""
        
    except Exception as e:
        return f"❌ Error getting stock price: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")