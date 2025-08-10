"""
AI Research Aggregator MCP Server
Aggregates research papers from arXiv, PubMed, and Semantic Scholar
No API keys needed - all free academic APIs!
"""

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json
import re
import urllib.parse

load_dotenv()

# Initialize MCP server
mcp = FastMCP("research-aggregator-server")

# API endpoints (all FREE, no keys needed!)
ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

@mcp.tool()
def search_arxiv_papers(query: str = "artificial intelligence", max_results: int = 5) -> str:
    """
    Search arXiv for research papers on any topic
    
    Args:
        query: Search query (e.g., "large language models", "computer vision", "reinforcement learning")
        max_results: Number of papers to return (default 5)
    """
    try:
        # Build arXiv query
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        response = requests.get(ARXIV_API, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()[:200]
            
            # Get authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
            
            # Get links
            pdf_link = ""
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                    break
            
            # Get published date
            published = entry.find('{http://www.w3.org/2005/Atom}published').text[:10]
            
            papers.append({
                'title': title,
                'authors': ', '.join(authors[:3]) + ('...' if len(authors) > 3 else ''),
                'summary': summary,
                'pdf': pdf_link,
                'date': published
            })
        
        if not papers:
            return f"❌ No papers found for '{query}' on arXiv"
        
        # Format results
        result = f"""📚 ARXIV RESEARCH PAPERS: {query.upper()}
        
🔬 Found {len(papers)} recent papers:

"""
        for i, paper in enumerate(papers, 1):
            result += f"""**{i}. {paper['title']}**
   👥 {paper['authors']}
   📅 {paper['date']}
   📝 {paper['summary']}...
   📄 [PDF]({paper['pdf']})
   
"""
        
        result += """💡 **Search Tips:**
• Try specific terms: "transformer architecture", "GPT", "BERT"
• Use author names: "Yann LeCun", "Geoffrey Hinton"
• Combine terms: "vision transformer medical imaging"
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error searching arXiv: {str(e)}"

@mcp.tool()
def search_pubmed_papers(query: str = "artificial intelligence medicine", max_results: int = 5) -> str:
    """
    Search PubMed for biomedical and life science research papers
    
    Args:
        query: Search query (e.g., "AI diagnosis", "machine learning cancer", "deep learning radiology")
        max_results: Number of papers to return (default 5)
    """
    try:
        # Step 1: Search for paper IDs
        search_url = f"{PUBMED_API}/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_response.json()
        
        id_list = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            return f"❌ No papers found for '{query}' on PubMed"
        
        # Step 2: Fetch paper details
        fetch_url = f"{PUBMED_API}/esummary.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(id_list),
            'retmode': 'json'
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
        fetch_data = fetch_response.json()
        
        papers = []
        for paper_id in id_list:
            paper_data = fetch_data.get('result', {}).get(paper_id, {})
            
            if paper_data:
                title = paper_data.get('title', 'No title')
                authors = paper_data.get('authors', [])
                author_names = ', '.join([a.get('name', '') for a in authors[:3]])
                if len(authors) > 3:
                    author_names += '...'
                
                journal = paper_data.get('source', 'Unknown journal')
                pub_date = paper_data.get('pubdate', 'Unknown date')
                
                papers.append({
                    'title': title,
                    'authors': author_names or 'Unknown authors',
                    'journal': journal,
                    'date': pub_date,
                    'pmid': paper_id,
                    'link': f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/"
                })
        
        # Format results
        result = f"""🧬 PUBMED BIOMEDICAL RESEARCH: {query.upper()}
        
🔬 Found {len(papers)} relevant papers:

"""
        for i, paper in enumerate(papers, 1):
            result += f"""**{i}. {paper['title']}**
   👥 {paper['authors']}
   📖 {paper['journal']}
   📅 {paper['date']}
   🔗 [View on PubMed]({paper['link']})
   
"""
        
        result += """💡 **Medical AI Topics:**
• "deep learning radiology"
• "machine learning diagnosis"
• "AI drug discovery"
• "computer vision pathology"
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error searching PubMed: {str(e)}"

@mcp.tool()
def search_semantic_scholar(query: str = "large language models", max_results: int = 5) -> str:
    """
    Search Semantic Scholar for academic papers with citation data
    
    Args:
        query: Search query (e.g., "transformers", "neural networks", "attention mechanism")
        max_results: Number of papers to return (default 5)
    """
    try:
        # Search Semantic Scholar
        url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors,abstract,year,citationCount,url,venue,publicationDate'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchAggregator/1.0)'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get('data', [])
        
        if not papers:
            return f"❌ No papers found for '{query}' on Semantic Scholar"
        
        # Format results
        result = f"""🎓 SEMANTIC SCHOLAR RESEARCH: {query.upper()}
        
📊 Found {len(papers)} papers with citation data:

"""
        for i, paper in enumerate(papers, 1):
            authors = paper.get('authors', [])
            author_names = ', '.join([a.get('name', '') for a in authors[:3]])
            if len(authors) > 3:
                author_names += '...'
            
            abstract = paper.get('abstract', 'No abstract available')
            if abstract and len(abstract) > 200:
                abstract = abstract[:200] + '...'
            
            citations = paper.get('citationCount', 0)
            year = paper.get('year', 'Unknown')
            venue = paper.get('venue', 'Unknown venue')
            url = paper.get('url', '#')
            
            # Determine impact level
            if citations > 1000:
                impact = "🔥 Highly Cited"
            elif citations > 100:
                impact = "⭐ Well Cited"
            elif citations > 10:
                impact = "📈 Growing Impact"
            else:
                impact = "🌱 Recent Paper"
            
            result += f"""**{i}. {paper.get('title', 'No title')}**
   👥 {author_names or 'Unknown authors'}
   📅 {year} | 📖 {venue}
   📊 Citations: {citations} | {impact}
   📝 {abstract}
   🔗 [View Paper]({url})
   
"""
        
        result += """💡 **Research Tips:**
• Sort by citations to find influential papers
• Look for recent papers (2023-2024) for latest developments
• Check venue (NeurIPS, ICML, ACL) for quality
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error searching Semantic Scholar: {str(e)}"

@mcp.tool()
def aggregate_ai_research(topic: str = "artificial intelligence", sources: str = "all") -> str:
    """
    Aggregate AI research from multiple sources (arXiv, PubMed, Semantic Scholar)
    
    Args:
        topic: Research topic to search
        sources: Which sources to use - 'all', 'arxiv', 'pubmed', 'semantic' (default 'all')
    """
    try:
        results = []
        
        # Determine which sources to search
        search_arxiv = sources.lower() in ['all', 'arxiv']
        search_pubmed = sources.lower() in ['all', 'pubmed']
        search_semantic = sources.lower() in ['all', 'semantic']
        
        result = f"""🔬 AI RESEARCH AGGREGATOR: {topic.upper()}
        
📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

"""
        
        # Search arXiv
        if search_arxiv:
            result += "📚 **arXiv Results:**\n"
            arxiv_results = search_arxiv_papers(topic, 3)
            if "❌" not in arxiv_results:
                result += "✅ Found papers on arXiv\n"
            else:
                result += "❌ No results on arXiv\n"
        
        # Search PubMed
        if search_pubmed:
            result += "\n🧬 **PubMed Results:**\n"
            pubmed_results = search_pubmed_papers(topic, 3)
            if "❌" not in pubmed_results:
                result += "✅ Found biomedical papers\n"
            else:
                result += "❌ No biomedical papers found\n"
        
        # Search Semantic Scholar
        if search_semantic:
            result += "\n🎓 **Semantic Scholar Results:**\n"
            semantic_results = search_semantic_scholar(topic, 3)
            if "❌" not in semantic_results:
                result += "✅ Found papers with citation data\n"
            else:
                result += "❌ No results on Semantic Scholar\n"
        
        result += f"""

📊 **Quick Stats:**
• Sources searched: {sources}
• Topic: {topic}
• Papers found: Check individual sources above

💡 **Next Steps:**
1. Use individual search tools for detailed results
2. Filter by year for recent papers
3. Sort by citations for influential work
4. Check PDF availability on arXiv
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error aggregating research: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")