import re
import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import subprocess
import sys
import tempfile
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
load_dotenv()


stopwords_file = "簡繁體停用字整合by_陳祐瑞.txt"
# GitHub 上該檔的原始 URL
stopwords_url = (
    "https://raw.githubusercontent.com/"
    "UrayChenNTHU/Integration-of-Simplified-and-Traditional-Chinese-Stop-words-by-Uray-Chen/"
    "refs/heads/main/簡繁體停用字整合by_陳祐瑞.txt"
)

def download_stopwords(url: str, target_path: str):
    resp = requests.get(url)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(resp.content)

# 「模組 import 時」先檢查本地檔案，若不存在就自動下載
if not os.path.exists(stopwords_file):
    try:
        download_stopwords(stopwords_url, stopwords_file)
        print(f"已從 GitHub 下載停用詞檔 → {stopwords_file}")
    except Exception as e:
        # 如果無法下載，也可以先用空集合，後續呼叫 is_stop() 時再決定
        print(f"⚠ 無法下載停用詞檔：{e}")

with open(stopwords_file, encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

import unicodedata
import jieba
import string
import zipfile
import base64
from typing import Optional

__all__ = [
    "fetch_full_page",
    "get_review_list",
    "scrape_udn_game_news_articles",
]


CONTENT_TAG_RE = re.compile(r"'content_tag':\s*\"([^\"]+)\"")
DATE_RE = re.compile(r"'publication_date':\s*'([^']+)'" )
chinese_punct = "…，。！？；：、“”‘’（）《》【】-"
punctuation_set = set(string.punctuation) | set(chinese_punct)
csv_file = "udn game corner game review article dataframe.csv"
zip_file = "udn-game-corner-game-review-article.zip"

def fetch_full_page(url: str, headless: bool = True, scroll_pause: float = 2.0) -> str:
    """
    使用 Selenium 獲取完整動態載入的頁面 HTML，包括滾動到底部。

    Args:
        url: 目標頁面 URL。
        headless: 是否以 headless 模式啟動瀏覽器。
        scroll_pause: 每次滾動後暫停秒數。

    Returns:
        完整的 HTML 字串。
    """
    print('0123456789')
    options = Options()
    options.add_argument("--headless")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--start-maximized")
    print('option')
    driver = Chrome(options=options)
    print('driver done')
    driver.get(url)

    # 滾動到底部以載入所有內容
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    html = driver.page_source
    driver.quit()
    return html


def get_review_list(category_url: str = 'https://game.udn.com/game/cate/122088', headless: bool = True) -> list[tuple]:
    """
    獲取 UDN-遊戲角落心得評測文章的 URL 與標題列表。

    Args:
        category_url: 分類頁面 URL（預設心得評測分類）。
        headless: 是否使用 headless 模式。

    Returns:
        List of (url, title) tuples。
    """
    html = fetch_full_page(category_url, headless=headless)
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('a', class_='story-list__link')

    reviews = []
    for a in items:
        slot = a.get('data-slotname')
        if slot == 'list_心得評測':
            href = a.get('href')
            title = a.get('title')
            if href and title:
                reviews.append((href, title))
    return reviews


def scrape_single_game_news_article(index: int, url: str, title: str) -> tuple[int, dict]:
    """
    爬取單篇心得評測文章內容。
    Returns a tuple of (index, article_dict).

    article_dict keys:
        url, title, author, author_description,
        publication_date (pd.Timestamp), topics, content
    """
    with requests.Session() as session:
        res = session.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

        # 作者與簡介
        author_el = soup.find('h3', class_='name')
        author = author_el.get_text(strip=True) if author_el else ''
        desc_el = soup.find('div', class_='context-box__text')
        author_desc = desc_el.get_text(' ', strip=True) if desc_el else ''

        # 文章內容段落
        paragraphs = [p for p in soup.find_all('p')]
        content = '\n'.join(p.get_text(' ', strip=True) for p in paragraphs)

        # 取得腳本中的 dataLayer 資訊
        script = soup.find('script', string=re.compile('dataLayer'))
        tag = ''
        pub_date = None
        if script and script.string:
            m = CONTENT_TAG_RE.search(script.string)
            tag = m.group(1) if m else ''
            d = DATE_RE.search(script.string)
            if d:
                pub_date = pd.to_datetime(d.group(1), errors='coerce')

        return index, {
            'url': url,
            'title': title,
            'author': author,
            'author_description': author_desc,
            'publication_date': pub_date,
            'topics': tag,
            'content': content
        }


def scrape_udn_game_news_articles(url_list: list[tuple], max_workers: int = 10) -> pd.DataFrame:
    """
    多執行緒爬取多篇文章，並以 DataFrame 回傳，保持原始順序。

    Args:
        url_list: list of (url, title)
        max_workers: 最大平行執行緒數量

    Returns:
        pandas.DataFrame: columns=['url','title','author',...]
    """
    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (url, title) in enumerate(url_list):
            futures.append(executor.submit(scrape_single_game_news_article, idx, url, title))

        for future in tqdm(as_completed(futures), total=len(futures), desc='爬取進度'):
            idx, art = future.result()
            results.append((idx, art))

    # 按照 index 排序，再轉成 DataFrame
    results.sort(key=lambda x: x[0])
    articles = [a for _, a in results]
    return pd.DataFrame(articles)

def clean_html_text(raw_text):
    no_ctrl = ''.join(ch for ch in raw_text if unicodedata.category(ch)[0] != 'C')
    no_zero_width = re.sub(r'[\u200B-\u200F\u202A-\u202E]', '', no_ctrl)
    normalized = re.sub(r'\s+', ' ', no_zero_width).strip()
    return normalized

def is_stop(token: str) -> bool:
    if token.strip() == "":               # 空字串或全空白
        return True
    # 如果 token 裡每個字元都在標點集合裡，就當作停用
    if all(ch in punctuation_set for ch in token):
        return True
    if token.isdigit():                   # 純數字（半形阿拉伯數字）
        return True
    if token in stopwords:                # 在停用詞列表裡
        return True
    return False


def udn_raw_df_preprocess(article_data):
    article_data['publication_date'] = article_data['publication_date'].bfill()
    article_data['remove control ch'] = article_data['content'].apply(lambda text: clean_html_text(text))
    article_data['tokenize'] = article_data['remove control ch'].apply(lambda text: jieba.lcut(text))
    article_data['tokenize and stop words'] = article_data['remove control ch'].apply(
        lambda text: [token for token in jieba.lcut(text) if not is_stop(token)]
    )
    article_data['tokenize and stop words without remove control'] = article_data['content'].apply(
        lambda text: [token for token in jieba.lcut(text) if not is_stop(token)]
    )

    # 使用正則表達式擷取《》內的內容
    article_data['game_name'] = article_data['title'].apply(lambda title: re.findall(r'《(.*?)》', title))
    article_data.to_csv('udn game corner game review article dataframe.csv', index=False,
        encoding='utf-8-sig')
    return article_data
def make_zip_from_csv() -> None:
    """
    將指定的 csv 檔壓縮成 zip。zip 內只包含同名的 csv 檔。
    例如：udn_game.csv → udn-game-corner-game-review-article.zip（內含 udn_game.csv）
    """

    csv_path = "udn game corner game review article dataframe.csv"
    zip_path = "udn-game-corner-game-review-article.zip"

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 注意：arcname 可以決定 zip 內檔案的名稱，這裡維持原檔名不變
        zf.write(csv_path, arcname=os.path.basename(csv_path))

def upload_udn_data_to_github(
    zip_path: str,
    repo_full_name: str = "UrayChenNTHU/udn-game-corner-game-review-article",
    target_path: str = "udn-game-corner-game-review-article.zip",
    commit_message: str = "Update UDN game review ZIP",
    github_token_env: str = "GITHUB_TOKEN"
) -> None:
    """
    把本地的 zip_path 上傳（或更新）到 GitHub Repo 指定位置：

      repo_full_name: 要更新的 Repo，例如 "UrayChenNTHU/udn-game-corner-game-review-article"
      target_path:    在 Repo 裡的相對路徑，例如 "udn-game-corner-game-review-article.zip"
      commit_message: 提交時的 commit 訊息
      github_token_env: 環境變數名稱，裡面存放 GitHub Personal Access Token

    如果該檔案已存在，就更新；若不存在，就新建。
    """
    # 1) 讀取 Personal Access Token
    token = os.getenv(github_token_env)
    if not token:
        raise RuntimeError(f"請先把你的 GitHub Token 設到環境變數 {github_token_env} 裡面。")
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 2) 讀取本地 ZIP 檔並 base64 編碼
    with open(zip_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 3) 先 GET 該檔案在 Repo 裡的資訊，看看是否已經存在（以取得 sha）
    api_url = f"https://api.github.com/repos/{repo_full_name}/contents/{target_path}"
    resp = requests.get(api_url, headers=headers)
    if resp.status_code == 200:
        # 檔案已存在，取得 sha 以便更新
        sha = resp.json().get("sha")
    elif resp.status_code == 404:
        sha = None
    else:
        raise RuntimeError(f"GitHub GET content 失敗：{resp.status_code} {resp.text}")

    # 4) 準備 PUT request 的 Payload
    data = {
        "message": commit_message,
        "content": content_b64,
        # 如果是更新，就帶上 sha；新增則省略
        **({"sha": sha} if sha else {})
    }

    put_resp = requests.put(api_url, headers=headers, json=data)
    if not (200 <= put_resp.status_code < 300):
        raise RuntimeError(f"GitHub PUT content 失敗：{put_resp.status_code}\n{put_resp.text}")

    action = "Updated" if sha else "Created"
    print(f"[Info] {action} {target_path} in {repo_full_name} successfully.")

def run_all():
    """
    一次執行完整的：爬 UDN 遊戲心得 → 前處理 → 輸出 CSV → 壓縮成 ZIP → 上傳 GitHub
    並回傳最終的 DataFrame 讓後續可以拿來做 TF-IDF、推薦系統等。
    """
    urls = get_review_list()
    
    df = scrape_udn_game_news_articles(urls)

    df = udn_raw_df_preprocess(df)

    make_zip_from_csv()

    upload_udn_data_to_github(zip_path="udn-game-corner-game-review-article.zip")

    return df

if __name__ == '__main__':
    # 簡易測試
    df=run_all()
    print(df.head())
    print('ALL Done')
