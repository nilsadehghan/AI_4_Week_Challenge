import requests
from bs4 import BeautifulSoup
import re
import time
import os
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def clean_extra_newlines(text):
    # Replace multiple newlines and surrounding whitespace with a single newline
    text = re.sub(r'[\s\u200c]*\n[\s\u200c]*', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)  # Ensure no double newlines left
    return text.strip()




def get_article_links(page_url):
    response = requests.get(page_url, headers=headers)
    time.sleep(2)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')

        links = []
        for article in articles:
            a_tag = article.find('a', href=True)
            if a_tag:
                link = a_tag['href']
                title = a_tag.get_text(strip=True)

                if link not in links:
                    links.append(link)
                    print(f'Title: {title}, Link: {link}')
        return links
    else:
        print(f'Failed to retrieve page, status code: {response.status_code}')
        return []

def get_article_content(article_url):
    response = requests.get(article_url, headers=headers)
    time.sleep(2)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        h1_tag = soup.find('h1')
        h1_text = h1_tag.get_text(strip=True) if h1_tag else "No title found."
        content_data = []
        content_div = soup.find('div', class_='post-inner')

        if content_div:


            # Remove all tables
            for table in content_div.find_all('table'):
                table.extract()

                # Remove video tags
            for unwanted_video in content_div.find_all('video'):
                unwanted_video.extract()

            # Extract first paragraph before any H2
            first_p_content = []
            for element in content_div.find_all(['p', 'h2']):
                if element.name == 'h2':
                    break
                if element.name == 'p':
                    first_p_content.append(element.get_text(strip=False))

            if first_p_content:
                content_data.append({
                    "heading1": h1_text,
                    "heading2": None,
                    "content": clean_extra_newlines("\n\n".join(first_p_content)),
                    "link": article_url
                })

            # Extract sections based on H2 tags
            for h2_tag in content_div.find_all('h2'):
                h2_text = h2_tag.get_text(strip=False)
                if not h2_text:
                    continue

                content = []
                for sibling in h2_tag.find_next_siblings():
                    if sibling.name == 'h2':
                        break

                    content.append(sibling.get_text(strip=False))

                if content:
                    content_data.append({
                        "heading1": h1_text,
                        "heading2": h2_text,
                        "content": clean_extra_newlines("\n\n".join(content)),
                        "link": article_url
                    })


            article_name = article_url.rstrip('/').split('/')[-1]
            filename = f"{article_name}.json"
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(content_data, file, ensure_ascii=False, indent=4)
            print(f'Content saved to {filename}')
    else:
        print(f'Failed to retrieve article, status code: {response.status_code}')


for n in range(1, 3):
    page_url = f"https://gadgetnews.net/page/{n}/"
    article_links = get_article_links(page_url)

    for link in article_links:
        get_article_content(link)