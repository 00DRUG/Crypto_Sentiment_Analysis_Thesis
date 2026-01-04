import requests
import json
from DatabaseWorker.db_manager import DatabaseManager
from Scrappers_legacy.Fetcher_deprecated import fetch_full_page

INDEXES_BY_YEAR = {
    2014: "CC-MAIN-2014-49",
    2015: "CC-MAIN-2015-48",
    2016: "CC-MAIN-2016-44",
    2017: "CC-MAIN-2017-47",
    2018: "CC-MAIN-2018-47",
    2019: "CC-MAIN-2019-47",
    2020: "CC-MAIN-2020-45",
    2021: "CC-MAIN-2021-49",
    2022: "CC-MAIN-2022-49",
    2023: "CC-MAIN-2023-40",
    2024: "CC-MAIN-2024-46",
}


class CommonCrawlScraper:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def search_commoncrawl(self, year, query):
        crawl_id = INDEXES_BY_YEAR[year]
        url = f"https://index.commoncrawl.org/{crawl_id}-index?url=*.{query}*&output=json"

        print(f"Searching year {year} at {url} ...")

        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            print(f"Failed to get data for {year}")
            return []

        lines = response.text.strip().split('\n')
        results = [json.loads(line) for line in lines]
        return results

    def scrape_and_store(self, keyword):
        for year in range(2014, 2025):
            self.db_manager.create_table(year)
            results = self.search_commoncrawl(year, keyword)

            saved_count = 0
            for result in results:
                warc_filename = result.get('filename')
                offset = int(result.get('offset', 0))
                length = int(result.get('length', 0))

                if not warc_filename or offset <= 0 or length <= 0:
                    continue

                full_page = fetch_full_page(warc_filename, offset, length)
                if full_page is None:
                    continue

                url = full_page.get('url', '')
                html = full_page.get('html', '')
                text = full_page.get('text', '')
                publish_date = result.get('timestamp', '')
                language = result.get('languages', '')

                # Filter: only English pages
                if language and 'eng' in language:
                    self.db_manager.save_result(year, url, '', text, html, publish_date, language)
                    saved_count += 1

            print(f"Saved {saved_count} full results for {year}")


