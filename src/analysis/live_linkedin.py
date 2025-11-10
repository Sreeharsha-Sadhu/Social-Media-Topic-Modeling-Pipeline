# live_linkedin.py

import asyncio
from pyppeteer import launch
import re

MIN_POST_REQ = 50


def extract_post_link(urn):
    """Helper to build a direct post link from a URN."""
    match = re.search(r"urn:li:activity:(\d+)", urn)
    if match:
        return f"https://www.linkedin.com/feed/update/urn:li:activity:{match[1]}"
    return "Post link not found"


async def scrape_linkedin_feed(username, password):
    """
    Scrapes LinkedIn feed and returns a list of post data.
    This function no longer uses Spark.
    """
    print("Launching browser...")
    browser = await launch(
        headless=True,
        executablePath="C:/Program Files/Google/Chrome/Application/chrome.exe",  # <-- change if needed
        args=["--no-sandbox"],
        defaultViewport=None
    )
    
    posts = []  # This will hold our scraped data
    
    try:
        page = await browser.newPage()
        await page.goto("https://www.linkedin.com/login")
        
        if await page.querySelector("#username"):
            print("Logging in to LinkedIn...")
            await page.type("#username", username, {"delay": 100})
            await page.type("#password", password, {"delay": 100})
            await page.click("button[type='submit']")
            await page.waitForNavigation()
            print("Login successful.")
        else:
            print("Already logged in, skipping login step.")
        
        await page.goto("https://www.linkedin.com/feed/")
        print("Waiting for feed to load...")
        await page.waitForSelector("div.feed-shared-update-v2")
        
        async def scroll_to_load_more_posts():
            previous_height = None
            while True:
                previous_height = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                print(f"Scrolling... {len(posts)} posts found so far.")
                await asyncio.sleep(2)
                new_height = await page.evaluate("document.body.scrollHeight")
                feed_handles = await page.querySelectorAll("div.feed-shared-update-v2")
                if new_height == previous_height or len(feed_handles) >= MIN_POST_REQ:
                    break
        
        await scroll_to_load_more_posts()
        
        feed_handles = await page.querySelectorAll("div.feed-shared-update-v2")
        number_of_posts = min(MIN_POST_REQ, len(feed_handles))
        print(f"Scraping details for {number_of_posts} posts...")
        
        for i in range(number_of_posts):
            feed_handle = feed_handles[i]
            
            # Name
            name = await page.evaluate(
                """(el) => {
                    let nameElement = el.querySelector("span > span > span:nth-child(1)") ||
                                      el.querySelector("span.update-components-actor__name.hoverable-link-text.t-14.t-bold.t-black.update-components-actor__single-line-truncate > span > span:nth-child(1)");
                    if (!nameElement) return "Name not found";
                    const fullName = nameElement.textContent.trim();
                    return fullName.slice(Math.floor(fullName.length / 2));
                }""", feed_handle
            )
            
            # Content
            feed = await page.evaluate(
                """(el) => {
                    let feedElement = el.querySelector("div.feed-shared-update-v2__description");
                    if (!feedElement) return "Content not found";
                    return feedElement.textContent.replace(/…more$/, '').trim();
                }""", feed_handle
            )
            
            # URN -> Post Link
            urn = await page.evaluate(
                """(el) => el.getAttribute('data-urn') || "URN not found"
                """, feed_handle
            )
            post_link = extract_post_link(urn)
            
            if name != "Name not found" and feed != "Content not found" and feed:
                posts.append({"name": name, "content": feed, "post_link": post_link})
        
        print(f"Successfully scraped {len(posts)} posts.")
    
    except Exception as e:
        print(f"--- 🚨 Error during LinkedIn scraping --- \nError: {e}")
    finally:
        await browser.close()
        print("Browser closed.")
        return posts  # Return the list of post dicts
