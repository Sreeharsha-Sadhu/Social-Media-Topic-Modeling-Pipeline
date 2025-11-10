# text_extractor.py

import praw
from requests_html import HTMLSession

# --- Setup for general web scraping ---
session = HTMLSession()


def extract_text_from_url(url):
    """
    Fetches a URL and extracts all paragraph text.
    This is the core function for your future web-scraping task.
    """
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        # Find all paragraph tags and join their text
        paragraphs = r.html.find('p')
        full_text = " ".join([p.text for p in paragraphs if p.text])
        
        if not full_text:
            print(f"  (No <p> text found at {url})")
            return None
        
        return full_text
    except Exception as e:
        print(f"  (Failed to scrape URL {url}: {e})")
        return None


# --- Setup for Reddit API ---
def get_reddit_client():
    """Initializes the PRAW Reddit client."""
    return praw.Reddit(
        client_id="jvwRh7vJI6ENPvtmtSIx7A",
        client_secret="GLl0aSeEjOKtIokHyIVBnMEd55U1AA",
        user_agent="windows:SocialAnalyzer:v1.0 by /u/sonilash",
    )


def get_reddit_post_content(post):
    """
    Extracts text content from a PRAW post object.
    If it's a link, it scrapes the destination URL.
    """
    # If it's a self-post, the text is in 'selftext'
    if post.is_self:
        return post.selftext
    
    # If it's a link post, scrape the URL
    url = post.url
    # Avoid scraping common image/video hosts
    if any(domain in url for domain in ['imgur.com', 'gfycat.com', 'redd.it', 'v.redd.it']):
        return None
    
    print(f"  Scraping link: {url[:50]}...")
    return extract_text_from_url(url)
