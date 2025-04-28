import requests
import gzip
import io
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup


def fetch_full_page(warc_filename, offset, length):
    base_url = "https://data.commoncrawl.org/"
    full_url = base_url + warc_filename

    headers = {
        "Range": f"bytes={offset}-{offset + length - 1}"
    }

    response = requests.get(full_url, headers=headers, timeout=60)
    if response.status_code != 206:
        print(f"Failed to fetch partial content: {response.status_code}")
        return None

    raw_data = io.BytesIO(response.content)
    stream = gzip.GzipFile(fileobj=raw_data)

    for record in ArchiveIterator(stream):
        if record.rec_type == 'response':
            url = record.rec_headers.get_header('WARC-Target-URI')
            payload = record.content_stream().read()

            # Decode payload (basic HTML)
            try:
                html_content = payload.decode('utf-8', errors='replace')
            except Exception as e:
                print(f"Decode error: {e}")
                return None

            # Extract visible text using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n')

            cleaned_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

            return {
                'url': url,
                'html': html_content,
                'text': cleaned_text
            }

    return None
