import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urljoin, urlparse
import extruct
import json
import PyPDF2
import io
import re
import pandas as pd

class WebsiteCrawlerSpider(CrawlSpider):
    name = 'web_crawler'
    start_urls = ['https://www.healthhub.sg']
    allowed_domains = ['www.healthhub.sg']
    def process_links(self, links):
        for link in links:
            if '#' not in link.url:  # Filter out links with '#'
                yield link

    rules = (
        Rule(LinkExtractor(), process_links='process_links', callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        # Extract metadata
        metadata = extruct.extract(response.text, response.url, syntaxes=["opengraph"])
        
        # Extract content
        content = []
        for element in response.css('p, span, table'):
            if element.root.tag == 'table':
                content.append(self.process_table(element))
            else:
                content.append(self.preprocess(element.xpath('string()').get()))


        # Extract .mp4 video links
        video_links = response.css('a[href$=".mp4"]::attr(href), video source[src$=".mp4"]::attr(src)').getall()
        
        
        # Extract .mp4 video links
        video_links = response.css('a[href$=".mp4"]::attr(href), video source[src$=".mp4"]::attr(src)').getall()
        
        yield {
            "name": self.get_title(metadata),
            "description": self.get_description(metadata),
            "response_type": "Website",
            "text": " ".join(content),
            "url": response.url
        }

        # Process PDF links
        pdf_links = response.css('a[href$=".pdf"]::attr(href)').getall()
        for pdf_link in pdf_links:
            yield scrapy.Request(
                urljoin(response.url, pdf_link),
                callback=self.parse_pdf,
                meta={'source_url': response.url}
            )

    def process_table(self, table):
        rows = []
        headers = [self.preprocess(header.xpath('string()').get()) for header in table.css('th')]
        
        if not headers:
            headers = [f"Column {i+1}" for i in range(len(table.css('tr:first-child td')))]
        
        for row in table.css('tr'):
            cells = [self.preprocess(cell.xpath('string()').get()) for cell in row.css('td')]
            if cells:
                row_dict = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
                rows.append(row_dict)
        
        table_str = "Table:\n"
        for row in rows:
            table_str += " | ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
        
        return table_str.strip()

    def parse_pdf(self, response):
        pdf_content = io.BytesIO(response.body)
        pdf_reader = PyPDF2.PdfReader(pdf_content)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        pdf_metadata = pdf_reader.metadata
        yield {
            "name": pdf_metadata.get('/Title', 'Untitled PDF'),
            "description": pdf_metadata.get('/Subject', ''),
            "response_type": "PDF",
            "text": text,
            "url": response.url,
            "source_url": response.meta['source_url']
        }

    def get_title(self, metadata):
        return metadata.get("opengraph", [{}])[0].get("title", "")

    def get_description(self, metadata):
        return metadata.get("opengraph", [{}])[0].get("description", "")

    def preprocess(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\$", " Dollar ", text)
        text = re.sub(r"\\n", "\n", text)
        text = bytes(text, "utf-8").decode("unicode_escape")
        text = re.sub(r"[^\x20-\x7E]+", "", text)
        return text.strip()