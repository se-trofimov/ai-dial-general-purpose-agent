import io
from pathlib import Path

import pandas as pd
import pdfplumber
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:
    def __init__(self, endpoint: str, api_key: str):
        self.client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        try:
            downloaded = self.client.files.download(file_url)
            return self.__extract_text(
                downloaded.get_content(), Path(downloaded.filename).suffix.lower(),
            )
        finally:
            self.client._http_client.internal_http_client.close()

    def __extract_text(self, file_content: bytes, file_extension: str) -> str:
        if file_extension == '.pdf':
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        text = file_content.decode('utf-8-sig')
        if file_extension == '.csv':
            return pd.read_csv(io.StringIO(text)).to_markdown(index=False)
        if file_extension in ('.html', '.htm'):
            soup = BeautifulSoup(text, 'html.parser')
            for element in soup(['script', 'style']):
                element.decompose()
            return soup.get_text(separator='\n', strip=True)
        if file_extension == '.txt':
            return text
        raise ValueError(f'Unsupported file extension: {file_extension}')
