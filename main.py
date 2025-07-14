from DatabaseWorker.db_manager import DatabaseManager
from Scrappers.crawl_scrapper import CommonCrawlScraper
import os
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    db_manager = DatabaseManager(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )

    scraper = CommonCrawlScraper(db_manager)
    scraper.scrape_and_store(keyword="bitcoin")

    db_manager.close()
