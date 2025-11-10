# live_twitter.py

import asyncio

import nest_asyncio
from pyppeteer import launch

# Allow nested event loops
nest_asyncio.apply()

# Configuration
MIN_POST_REQ = 30
HEADLESS_MODE = False  # Set to False to see the browser


async def scrape_feed(feed_handles, page, collected_posts, existing_posts):
    """Scrape each tweet element and append unique ones."""
    for feed_handle in feed_handles:
        # --- Extract name ---
        name = await page.evaluate("""
            el => {
                const nameEl = el.querySelector('div[dir="ltr"] span span');
                return nameEl ? nameEl.textContent.trim() : "Name not found";
            }
        """, feed_handle)
        
        # --- Extract content ---
        content = await page.evaluate("""
            el => {
                const textEl = el.querySelector('div[data-testid="tweetText"]');
                return textEl ? textEl.textContent.trim() : "Content not found";
            }
        """, feed_handle)
        
        # --- Extract post link ---
        post_link = await page.evaluate("""
            el => {
                const linkEl = el.querySelector('a[href*="/status/"]');
                return linkEl ? linkEl.href : null;
            }
        """, feed_handle)
        
        # --- Duplicate check ---
        unique_key = (name, content, post_link)
        if unique_key in existing_posts or not content or content == "Content not found":
            continue
        
        collected_posts.append({"name": name, "content": content, "post_link": post_link})
        existing_posts.add(unique_key)
        
        print(f"  > Scraped post {len(collected_posts)} by: {name}")
    
    return len(collected_posts)


async def scrape_twitter_feed(username, password):
    """
    Scrapes Twitter feed and returns a list of post data.
    This function no longer uses Spark.
    """
    posts = []
    existing = set()
    
    try:
        print("Launching browser...")
        browser = await launch(
            headless=HEADLESS_MODE,
            executablePath="C:/Program Files/Google/Chrome/Application/chrome.exe",
            args=["--no-sandbox"],
            defaultViewport=None,
            userDataDir=None
        )
        
        page = await browser.newPage()
        print("Navigating to Twitter login page...")
        await page.goto("https://x.com/login", {"waitUntil": "networkidle2"})
        await asyncio.sleep(3)
        
        print("Logging in to Twitter/X...")
        await page.type('input[name="text"]', username, {"delay": 100})
        await page.keyboard.press("Enter")
        await asyncio.sleep(2)
        
        await page.type('input[name="password"]', password, {"delay": 100})
        await page.keyboard.press("Enter")
        await asyncio.sleep(5)
        print("Login successful.")
        
        print("Navigating to home timeline...")
        await page.goto("https://x.com/home", {"waitUntil": "networkidle2"})
        
        try:
            await page.waitForSelector('article div[data-testid="tweet"]', {'timeout': 60000})
            print("✅ Home timeline loaded. Ready to scrape posts.")
        except Exception:
            html = await page.content()
            print("❌ Could not find tweets. Dumping HTML for debugging:")
            print(html[:2000])
            raise
        
        async def scroll_to_load_more():
            previous_height = None
            attempts = 0
            max_attempts = 10
            
            while attempts < max_attempts and len(posts) < MIN_POST_REQ:
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await asyncio.sleep(2)
                new_height = await page.evaluate('document.body.scrollHeight')
                
                if new_height == previous_height:
                    print("No new posts loaded, attempt {attempts+1}/{max_attempts}")
                    attempts += 1
                else:
                    attempts = 0  # Reset on success
                
                feed_handles = await page.querySelectorAll('article div[data-testid="tweet"]')
                await scrape_feed(feed_handles, page, posts, existing)
        
        await scroll_to_load_more()
        
        print(f"Successfully scraped {len(posts)} posts.")
    
    except Exception as e:
        print(f"--- 🚨 Error during Twitter scraping --- \nError: {e}")
    finally:
        await browser.close()
        print("Browser closed.")
        return posts
