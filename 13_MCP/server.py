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
            "x-api-key": METALS_API_KEY,
            "Content-Type": "application/json"
        }
        
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
        # Get latest price
        result = make_metals_request(f"spot/{metal.lower()}", {"currency": currency.upper()})
        
        if "error" in result:
            return f"❌ Error fetching {metal} price: {result['error']}"
        
        # Parse response
        price = result.get("price", 0)
        change_24h = result.get("change_24h", 0)
        change_pct = result.get("change_percentage_24h", 0)
        timestamp = result.get("timestamp", "")
        
        # Determine trend emoji
        trend = "📈" if change_24h > 0 else "📉" if change_24h < 0 else "➡️"
        
        result_text = f"""🏆 {metal.upper()} SPOT PRICE
        
💰 Current Price: {currency} {price:,.2f} per oz
{trend} 24h Change: {change_24h:+.2f} ({change_pct:+.2f}%)

📊 Market Stats:
  • High (24h): {currency} {result.get('high_24h', price):,.2f}
  • Low (24h): {currency} {result.get('low_24h', price):,.2f}
  • Volume (24h): {result.get('volume_24h', 'N/A')} oz

⏰ Last Updated: {timestamp}

💡 Tip: Gold is traditionally seen as a safe-haven asset
📝 Note: Prices are for spot market (immediate delivery)"""
        
        return result_text
        
    except Exception as e:
        return f"❌ Error getting {metal} price: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")