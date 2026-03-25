"""Wikipedia Random Page node for ACE-Step"""
import random
import requests
import wikipediaapi
import logging

logger = logging.getLogger(__name__)

# Categories extracted from https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories
# Including top-level and immediate sub-categories per user request.
WIKI_CATEGORIES = [
    "General reference",
    "Culture and the arts",
    "Geography and places", 
    "Health and fitness",
    "History and events",
    "Human activities", 
    "Mathematics and logic",
    "Natural and physical sciences",
    "People and self", 
    "Philosophy and thinking",
    "Religion and belief systems",
    "Society and social sciences", 
    "Technology and applied sciences",
    # Sub-categories
    "Books",
    "Information",
    "Knowledge",
    "Lists",
    "Medical manuals",
    "Reference works",
    "Research",
    "Writing",
    "Architecture",
    "Arts",
    "Classics",
    "Clothing",
    "Comics",
    "Cultural anthropology",
    "Culture",
    "Dance",
    "Design",
    "Drawing",
    "Entertainment", 
    "Film",
    "Folklore",
    "Food and drink",
    "Games",
    "Humanities",
    "Language", 
    "Literature",
    "Mass media",
    "Museology",
    "Music",
    "Mythology",
    "Opera", 
    "Painting",
    "Performing arts",
    "Philosophy",
    "Photography",
    "Popular culture", 
    "Recreation",
    "Sculpture",
    "Sports",
    "Theatre",
    "Traditions",
    "Visual arts",
    "Cities",
    "Communities",
    "Continents",
    "Countries",
    "Earth",
    "World",
    "Diseases",
    "Genetics",
    "Health care",
    "Medicine",
    "Nutrition", 
    "Psychiatry",
    "Public health",
    "Empires",
    "Events",
    "Historiography",
    "History",
    "Timelines",
    "Business",
    "Education",
    "Industry", 
    "Law",
    "Leisure",
    "Military",
    "Politics",
    "Science",
    "Transport",
    "Travel",
    "Work",
    "Algebra",
    "Analysis",
    "Arithmetic",
    "Computer science",
    "Discrete mathematics", 
    "Geometry",
    "Logic",
    "Number theory",
    "Probability",
    "Statistics",
    "Topology", 
    "Trigonometry",
    "Animals",
    "Astronomy",
    "Chemistry",
    "Climate",
    "Environment",
    "Humans",
    "Life", 
    "Nature",
    "Physical sciences",
    "Physics",
    "Plants",
    "Scientific method", 
    "Scientists",
    "Space",
    "Universe",
    "Human body",
    "Human development",
    "Human nature",
    "Minds", 
    "Personality",
    "Self",
    "Senses",
    "Virtues",
    "Aesthetics",
    "Cognition",
    "Creativity",
    "Decision making",
    "Epistemology", 
    "Ethics",
    "Learning",
    "Memory",
    "Metaphysics",
    "Perception",
    "Problem solving", 
    "Psychology",
    "Social philosophy",
    "Belief systems", 
    "Anthropology",
    "Archaeology",
    "Business",
    "Crime",
    "Cultural studies", 
    "Economics",
    "Ethnic groups",
    "Family",
    "Finance",
    "Globalization",
    "Government", 
    "Infrastructure",
    "Money", 
    "Sociology",
    "Artificial intelligence",
    "Automation",
    "Biotechnology",
    "Cartography", 
    "Chemical engineering",
    "Computing",
    "Construction",
    "Databases",
    "Design", 
    "Engineering",
    "Internet", 
    "Management",
    "Manufacturing",
    "Marketing",
    "Nanotechnology",
    "Nuclear technology",
    "Robotics",
    "Software", 
    "Telecommunications"
]

class WikipediaRandomNode:
    """
    Fetches a random Wikipedia page's main content with optional filtering.
    Modes:
    - Truly Random: Global random page.
    - Word Search: Search keywords and pick random result from top 50.
    - Category Search: Pick random page from a specific category.
    
    Inputs:
        mode (STRING): Search methodology.
        category (STRING): Target category if in Category Search mode.
        search_keyword (STRING): Target keyword if in Word Search mode.
        language (STRING): Wikipedia subdomain (e.g., "en").
        user_agent (STRING): API identification header.
        seed (INT): Deterministic RNG seed.
        clean_content (BOOLEAN): Strips footers/references.
        
    Outputs:
        title (STRING): Found page title.
        content (STRING): Page body text.
        url (STRING): Direct link to article.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Truly Random", "Word Search", "Category Search"], {"default": "Truly Random"}),
                "category": (list(dict.fromkeys(WIKI_CATEGORIES)), {"default": "Music"}),
                "search_keyword": ("STRING", {"default": "", "multiline": False, "placeholder": "Keyword for Word Search mode"}),
                "language": (["en", "simple", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru"], {"default": "en"}),
                "user_agent": ("STRING", {"default": "ScromfyAce-Step/1.5 (comfyui)", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "clean_content": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("title", "content", "url")
    FUNCTION = "fetch"
    CATEGORY = "Scromfy/Ace-Step/Misc"

    def fetch(self, mode, category, search_keyword, language, user_agent, seed, clean_content=True):
        rng = random.Random(seed)
        wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )

        title = ""
        base_url = f"https://{language}.wikipedia.org/w/api.php"
        headers = {"User-Agent": user_agent}

        try:
            if mode == "Word Search":
                if not search_keyword.strip():
                    return ("Please provide a search keyword", "", "")
                
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": search_keyword,
                    "format": "json",
                    "srlimit": 50
                }
                resp = requests.get(base_url, params=params, headers=headers, timeout=10)
                data = resp.json()
                results = data.get("query", {}).get("search", [])
                
                if not results:
                    return (f"No results for: {search_keyword}", "", "")
                
                picked = rng.choice(results)
                title = picked["title"]

            elif mode == "Category Search":
                # Uses the prioritized category search logic
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{category}",
                    "cmlimit": 500,
                    "cmtype": "page"
                }
                resp = requests.get(base_url, params=params, headers=headers, timeout=10)
                data = resp.json()
                pages = data.get("query", {}).get("categorymembers", [])
                
                if not pages:
                    # Fallback: try searching for the category if no direct members found (meta-category)
                    return (f"No direct pages in Category:{category}. Try a more specific sub-category.", "", "")
                
                picked = rng.choice(pages)
                title = picked["title"]

            else:  # Truly Random
                params = {
                    "action": "query",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": 1,
                    "format": "json"
                }
                resp = requests.get(base_url, params=params, headers=headers, timeout=10)
                data = resp.json()
                results = data.get("query", {}).get("random", [])
                
                if not results:
                    return ("Error fetching random page", "", "")
                
                title = results[0]["title"]

            # Use wikipedia-api to get structured content
            page = wiki.page(title)
            
            if not page.exists():
                return (f"Page not found: {title}", "", "")

            # Combine summary and sections for "main content"
            # wikipedia-api's page.text usually starts with the summary, so we use it directly
            # to avoid duplication reported by users.
            content = page.text
            
            if clean_content:
                # Footer sections to truncate
                stop_headers = [
                    "== See also ==",
                    "== References ==",
                    "== Further reading ==",
                    "== External links ==",
                    "== Related pages ==",
                    "== Notes ==",
                    "== Sources ==",
                    "== Citations ==",
                    "== Bibliography =="
                ]
                
                # Also check for plain text versions as fallback
                plain_stops = [
                    "\nRelated pages\n",
                    "\nReferences\n",
                    "\nSee also\n"
                ]
                
                earliest_index = len(content)
                
                # Check structured headers
                for header in stop_headers:
                    idx = content.find(header)
                    if idx != -1 and idx < earliest_index:
                        earliest_index = idx
                
                # Check plain text fallbacks
                for stop in plain_stops:
                    idx = content.find(stop)
                    if idx != -1 and idx < earliest_index:
                        earliest_index = idx
                        
                if earliest_index < len(content):
                    content = content[:earliest_index].strip()

            return (page.title, content, page.fullurl)

        except Exception as e:
            logger.error(f"Wikipedia node error: {e}")
            return (f"Error: {str(e)}", "", "")

NODE_CLASS_MAPPINGS = {
    "WikipediaRandomNode": WikipediaRandomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WikipediaRandomNode": "Wikipedia Random Page",
}
