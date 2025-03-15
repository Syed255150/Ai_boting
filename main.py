from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
import re

# Proxy configuration
PROXY_URL = "https://brd-customer-hl_a75066f0-zone-scrapingwr:t1fr8cusr4iq@brd.superproxy.io:9515"

# List of User-Agent strings to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
]

def scrape_website(website):
    """
    Scrapes the given website using the provided proxy configuration and anti-bot measures.
    """
    print("Connecting to Scraping Browser...")

    # Rotate User-Agent to mimic different browsers
    user_agent = random.choice(USER_AGENTS)

    # Configure ChromeOptions with proxy, user-agent, and other anti-bot measures
    chrome_options = ChromeOptions()
    chrome_options.add_argument(f"--proxy-server={PROXY_URL}")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={user_agent}")

    # Establish connection to the remote driver
    sbr_connection = ChromiumRemoteConnection(
        "http://localhost:8501/wd/hub", "goog", "chrome"
    )  # Adjust URL if needed

    try:
        with Remote(sbr_connection, options=chrome_options) as driver:
            # Navigate to the website
            print(f"Navigating to {website}...")
            driver.get(website)

            # Introduce random delay to mimic human browsing
            time.sleep(random.uniform(2, 5))

            # Wait for page content to load (adjust timeout as necessary)
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception:
                print("Timeout waiting for page to load. The site may be blocking access.")
                return None

            # Check for CAPTCHA in the page source
            if "captcha" in driver.page_source.lower():
                print("CAPTCHA detected. Solve it manually if it appears...")
                time.sleep(30)  # Adjust time as needed for manual solving

            print("Page loaded. Scraping content...")

            # Extract the page source
            html = driver.page_source
            print("Scraping complete!")
            return html

    except Exception as e:
        print(f"An error occurred while scraping the website: {e}")
        return None

def extract_data_based_on_prompt(soup, prompt):
    """
    Extract data from the webpage based on the user's prompt
    """
    data_points = []
    prompt = prompt.lower()
    
    # Common patterns for different types of data
    patterns = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
        'address': r'\d{1,5}\s\w.\s(\b\w*\b\s){1,}\w*\.'
    }
    
    # Extract all text content
    all_text = soup.get_text()
    results = []
    
    if 'name' in prompt:
        names = [tag.text.strip() for tag in soup.find_all(['h1', 'h2', 'h3', 'strong']) if tag.text.strip()]
        results.extend(names[:10])
        
    if 'email' in prompt:
        emails = re.findall(patterns['email'], all_text)
        results.extend(emails)
        
    if 'phone' in prompt or 'contact' in prompt:
        phones = re.findall(patterns['phone'], all_text)
        results.extend(phones)
        
    if 'address' in prompt:
        addresses = re.findall(patterns['address'], all_text)
        results.extend(addresses)
    
    # Get all paragraphs if specific data isn't found or for general content
    if not results:
        results = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
    
    return results[:100]  # Return up to 100 results

def scrape_website_streamlit(url, prompt):
    """
    Scrape the website based on the given URL and prompt
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = extract_data_based_on_prompt(soup, prompt)
        
        return data
    except Exception as e:
        return [f"Error: {str(e)}"]

# Streamlit UI
st.set_page_config(page_title="AI Web Scraper", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Web Scraper")
st.markdown("""
This web scraper can extract information from websites based on your requirements.
Just enter the website URL and describe what you want to extract!
""")

# Input fields
url = st.text_input("Enter Website URL", placeholder="https://example.com")
prompt = st.text_area("What would you like to extract?", 
                     placeholder="Example: Extract names, email addresses, phone numbers, etc.")

# Start button with spinner
if st.button("Start Scraping"):
    if url and prompt:
        with st.spinner("Scraping data... Please wait"):
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            try:
                results = scrape_website_streamlit(url, prompt)
                
                # Display results
                st.success(f"Found {len(results)} results!")
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(results, columns=['Extracted Data'])
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "scraped_data.csv",
                    "text/csv",
                    key='download-csv'
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter both a URL and describe what you want to extract.")
