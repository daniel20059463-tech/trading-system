"""
新聞爬蟲模組：抓取 Yahoo Finance 和 MoneyDJ 的台股新聞。
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import xml.etree.ElementTree as ET
import urllib.parse

import requests
from bs4 import BeautifulSoup
import yfinance as yf

# 確保可以從專案根目錄執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 目標股票清單
TARGET_TICKERS = {
    "2327": "國巨",
    "2492": "華新科",
    "3026": "禾伸堂",
    "2472": "立隆電",
    "3357": "臺慶科",
    "5425": "台半",
    "2481": "強茂",
    "3675": "德微",
    "8255": "朋程",
    "8261": "富鼎",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NEWS_MAX_AGE = timedelta(days=7)


def parse_published_at(value):
    """新聞發布時間轉 UTC；未知時間不推定為盤前可用。"""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                parsed = datetime.strptime(text.replace("/", "-"), "%Y-%m-%d")
        if parsed.tzinfo is None:
            # 日期但無時間／時區的新聞無法證明在盤前發布。
            return None
        return parsed.astimezone(timezone.utc)
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def fresh_articles(articles, now=None, max_age=NEWS_MAX_AGE):
    """只讓已發布、七天內、時間可核對的文章進入盤前分析。"""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("現在時間必須含時區")
    now = now.astimezone(timezone.utc)
    return [article for article in articles
            if (published := parse_published_at(article.get("timestamp"))) is not None
            and now - max_age <= published <= now]


def _yfinance_news(yf_ticker: str, max_articles: int, source_tag: str) -> list[dict]:
    """用 yfinance .news 屬性抓新聞，回傳統一格式的清單。"""
    articles = []
    try:
        ticker_obj = yf.Ticker(yf_ticker)
        raw_news = ticker_obj.news or []
        for item in raw_news[:max_articles]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            if not title:
                continue
            url = ""
            for link in content.get("canonicalUrl", {}).values():
                if isinstance(link, str) and link.startswith("http"):
                    url = link
                    break
            pub_date = content.get("pubDate", "") or item.get("providerPublishTime", "")
            articles.append({
                "title": title,
                "url": url,
                "timestamp": str(pub_date) if pub_date else "",
                "source": source_tag,
            })
    except Exception as e:
        logger.debug(f"yfinance news ({yf_ticker})：{e}")
    return articles


def scrape_yahoo_news(ticker_tw: str, max_articles: int = 10) -> list[dict]:
    """用 yfinance 抓取個股新聞（ticker_tw 為純代碼如 2327，自動補 .TW/.TWO）。"""
    from config import ALL_STOCKS
    # 找出此代碼對應的完整 yfinance ticker
    yf_ticker = next(
        (k for k in ALL_STOCKS if k.startswith(ticker_tw + ".")),
        ticker_tw + ".TW",
    )
    articles = _yfinance_news(yf_ticker, max_articles, "yahoo")
    logger.info(f"Yahoo {ticker_tw}：抓到 {len(articles)} 篇新聞")
    return articles


def scrape_google_news(ticker_tw: str, company_name: str, max_articles: int = 10) -> list[dict]:
    """用 Google News RSS 抓取個股中文新聞，適合 yfinance 覆蓋不足的小型台股。"""
    query = urllib.parse.quote(f"{company_name} {ticker_tw}")
    url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    articles = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item")[:max_articles]:
            title_el = item.find("title")
            link_el  = item.find("link")
            date_el  = item.find("pubDate")
            source_el = item.find("source")
            if title_el is None or not title_el.text:
                continue
            articles.append({
                "title":     title_el.text.strip(),
                "url":       link_el.text.strip() if link_el is not None and link_el.text else "",
                "timestamp": date_el.text.strip() if date_el is not None and date_el.text else "",
                "source":    source_el.text if source_el is not None else "google_news",
            })
        logger.info(f"Google News {ticker_tw}（{company_name}）：抓到 {len(articles)} 篇新聞")
    except Exception as e:
        logger.warning(f"Google News 抓取失敗（{ticker_tw}）：{e}")
    return articles


def scrape_stock_news(ticker_tw: str, max_articles: int = 8) -> list[dict]:
    """個股新聞：中文 Google News 為主（在地題材如漲價/訂單/季報最相關），
    英文 yfinance 為輔，去重後合併。

    背景：yfinance 對國巨等大型股常只回傳英文國際併購新聞（如 Yageo 收購案），
    漏掉中文的漲價/訂單外溢題材；故改以中文 Google News 為主要來源。
    """
    from config import ALL_STOCKS
    full = next((k for k in ALL_STOCKS if k.startswith(ticker_tw + ".")), None)
    name = ALL_STOCKS.get(full, ticker_tw) if full else ticker_tw

    google = scrape_google_news(ticker_tw, name, max_articles=max_articles)
    yahoo = scrape_yahoo_news(ticker_tw, max_articles=4)   # 英文補充
    combined = {}
    for a in google + yahoo:                                # 中文優先
        t = a.get("title", "").strip()
        if t and t not in combined:
            combined[t] = a
    return fresh_articles(list(combined.values()))[:max_articles]


def scrape_peer_news(keywords: list[str], max_per_keyword: int = 3) -> list[dict]:
    """用 Google News RSS 搜尋同業關鍵字（村田/ON Semi/MLCC 缺貨等），回傳去重後新聞。

    用於偵測台股個股新聞未涵蓋的國際同業重大事件（財報、缺貨、擴產）。
    """
    seen = set()
    articles = []
    for kw in keywords:
        query = urllib.parse.quote(kw)
        url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:max_per_keyword]:
                title_el = item.find("title")
                date_el  = item.find("pubDate")
                src_el   = item.find("source")
                if title_el is None or not title_el.text:
                    continue
                title = title_el.text.strip()
                if title in seen:
                    continue
                seen.add(title)
                articles.append({
                    "title":     title,
                    "url":       "",
                    "timestamp": date_el.text.strip() if date_el is not None and date_el.text else "",
                    "source":    (src_el.text if src_el is not None else "google_news") + f"｜{kw}",
                })
        except Exception as e:
            logger.debug(f"同業新聞搜尋失敗（{kw}）：{e}")
        time.sleep(0.5)
    recent = fresh_articles(articles)
    logger.info(f"同業關鍵字新聞：抓到 {len(recent)} 篇可確認時間的近期新聞")
    return recent


def scrape_moneydj_news(ticker_tw: str, max_articles: int = 10) -> list[dict]:
    """抓取 MoneyDJ 的個股相關新聞列表。"""
    url = f"https://www.moneydj.com/IFRS/News/NewsM0002.djhtm?a={ticker_tw}"
    articles = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        seen_titles = set()

        # MoneyDJ 新聞通常在 table 或 ul 結構中
        rows = soup.select("table tr") or soup.select("ul.news-list li") or soup.select("div.news a")

        for row in rows[:max_articles * 3]:
            try:
                anchor = row.select_one("a") if row.name != "a" else row
                if not anchor:
                    continue

                title = anchor.get_text(strip=True)
                href = anchor.get("href", "")

                if not title or len(title) < 5 or title in seen_titles:
                    continue
                seen_titles.add(title)

                if href.startswith("/"):
                    href = "https://www.moneydj.com" + href
                elif not href.startswith("http"):
                    href = ""

                # 嘗試抓同行的時間文字
                timestamp = ""
                td_list = row.select("td")
                for td in td_list:
                    text = td.get_text(strip=True)
                    if len(text) in (8, 10) and ("/" in text or "-" in text):
                        timestamp = text
                        break

                articles.append({
                    "title": title,
                    "url": href,
                    "timestamp": timestamp,
                    "source": "moneydj",
                })

                if len(articles) >= max_articles:
                    break

            except Exception as e:
                logger.debug(f"解析 MoneyDJ 項目時發生錯誤：{e}")
                continue

        logger.info(f"MoneyDJ {ticker_tw}：抓到 {len(articles)} 篇新聞")

    except requests.RequestException as e:
        logger.warning(f"MoneyDJ 請求失敗（{ticker_tw}）：{e}")
    except Exception as e:
        logger.warning(f"MoneyDJ 解析錯誤（{ticker_tw}）：{e}")

    return articles


def get_market_news(max_articles: int = 15) -> list[dict]:
    """用 yfinance 抓取台股大盤新聞（^TWII 指數）。"""
    articles = _yfinance_news("^TWII", max_articles, "yahoo_market")
    logger.info(f"大盤新聞：抓到 {len(articles)} 篇")
    return articles


def fetch_all_news(tickers_list: list[str]) -> dict:
    """對所有指定股票抓取新聞：yfinance 為主，Google News RSS 為補充，去重後加入大盤新聞。"""
    from config import ALL_STOCKS

    result = {}

    for ticker in tickers_list:
        logger.info(f"開始抓取 {ticker} 的新聞...")

        # 中文 Google News 為主（在地題材：漲價/訂單/季報最相關），一律抓取
        full_ticker = next((k for k in ALL_STOCKS if k.startswith(ticker + ".")), None)
        company_name = ALL_STOCKS.get(full_ticker, ticker) if full_ticker else ticker
        google_news = scrape_google_news(ticker, company_name, max_articles=10)
        time.sleep(0.8)
        yahoo_news = scrape_yahoo_news(ticker, max_articles=6)   # 英文補充
        time.sleep(0.8)
        moneydj_news = scrape_moneydj_news(ticker, max_articles=6)
        time.sleep(0.8)

        combined = {}
        for article in google_news + yahoo_news + moneydj_news:   # 中文優先
            title = article.get("title", "").strip()
            if title and title not in combined:
                combined[title] = article

        result[ticker] = fresh_articles(list(combined.values()))
        logger.info(f"{ticker} 合併後共 {len(result[ticker])} 篇新聞"
                    + (" (含Google News)" if google_news else ""))

    logger.info("開始抓取大盤新聞...")
    market_news = get_market_news(max_articles=15)
    result["market"] = fresh_articles(market_news)

    return result


def save_news_to_json(news_data: dict, output_path: str) -> None:
    """將新聞資料存成 JSON 檔案。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)
    logger.info(f"新聞已存至：{output_path}")


if __name__ == "__main__":
    print("=== 測試爬取 2327（國巨）新聞 ===")

    yahoo = scrape_yahoo_news("2327", max_articles=5)
    print(f"\nYahoo Finance（共 {len(yahoo)} 篇）：")
    for i, a in enumerate(yahoo, 1):
        print(f"  {i}. [{a['source']}] {a['title']}")
        if a["url"]:
            print(f"     URL: {a['url'][:80]}...")
        if a["timestamp"]:
            print(f"     時間: {a['timestamp']}")

    time.sleep(1)

    moneydj = scrape_moneydj_news("2327", max_articles=5)
    print(f"\nMoneyDJ（共 {len(moneydj)} 篇）：")
    for i, a in enumerate(moneydj, 1):
        print(f"  {i}. [{a['source']}] {a['title']}")

    time.sleep(1)

    market = get_market_news(max_articles=5)
    print(f"\n大盤新聞（共 {len(market)} 篇）：")
    for i, a in enumerate(market, 1):
        print(f"  {i}. [{a['source']}] {a['title']}")

    print("\n=== 測試完成 ===")
