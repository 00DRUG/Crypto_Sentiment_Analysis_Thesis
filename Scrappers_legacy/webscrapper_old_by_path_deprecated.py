import time
from telnetlib3  import EC

import requests
import spacy
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait

# modules import
from Json import JsonLoader

from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")

links_file = '../Json/links.json'
user_agents_file = '../Json/user_agents.json'
keywords_file = '../Json/keywords.json'
DBWorker.create_predictions()


def search_articles(query, user_agents, one_link):
    browser_category = random.choice(list(user_agents.keys()))
    user_agent = random.choice(user_agents[browser_category])

    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)
    driver.get(one_link)

    try:
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
    except:
        print("Search box not found")
        print(driver.page_source)
        driver.quit()
        return []
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    all_links = []
    while True:
        time.sleep(3)

        articles = driver.find_elements(By.CSS_SELECTOR, ".css-14rwwjy-PromoContent a")
        for article in articles:
            link = article.get_attribute("href")
            if link and "/articles/" in link:
                all_links.append(link)

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, ".pagination-next-button")
            next_button.click()
            print("Loading next page...")
        except:
            print("No more pages to scrape.")
            break

    driver.quit()
    return all_links


def scrape_data(url, user_agents, data):
    for attempt in range(3):
        browser_category = random.choice(list(user_agents.keys()))
        user_agent = random.choice(user_agents[browser_category])
        headers = {'User-Agent': user_agent}

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                website_id = next((key for key, value in data.items() if value == url), None)
                print(f"Scraping data from {url} with User-Agent: {user_agent}")

                soup.find_all('article')
                text = soup.get_text(strip=True)
                return text, website_id

            elif response.status_code == 403:
                print(f"403 Forbidden error encountered. Retrying with a different User-Agent...")
            else:
                print(f"Failed to retrieve {url}: {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return []

    print(f"Failed to scrape {url} after 3 attempts.")
    return []


def main():
    DBWorker.create_predictions()
    user_agents = JsonLoader.load_json(user_agents_file)
    data = JsonLoader.load_json(links_file)
    for key, search_url in data.items():
        links = search_articles("Bitcoin", user_agents, search_url)
        for link in links:

            DBWorker.save_predictions(*scrape_data(links_file, user_agents, link))
        else:
            print("No articles found. Exiting.")
            return


if __name__ == "__main__":
    main()
