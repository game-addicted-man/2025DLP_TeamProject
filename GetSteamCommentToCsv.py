import requests
import csv
import time
import urllib.parse
import re
from bs4 import BeautifulSoup

# 1) 한국 스토어 기준 인기 게임(appid) 리스트 가져오기
def get_top_games(n_games, language='korean', sleep_sec=1.0):
    """
    Steam KR 스토어에서 Top Sellers 순으로 n_games개의 앱ID를 가져옵니다.
    - cc=kr: 대한민국 스토어
    - filter=topsellers: 판매량 기준 인기순
    - supportedlang=korean: 한글 인터페이스 게임 위주(선택)
    """
    base_url = 'https://store.steampowered.com/search/'
    params = {
        'cc': 'kr',
        'filter': 'topsellers',
        'supportedlang': language,
        'page': 1
    }
    games = []
    seen = set()

    while len(games) < n_games:
        resp = requests.get(base_url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = soup.find_all('a', class_='search_result_row')
        if not rows:
            break

        for row in rows:
            appid = row.get('data-ds-appid')
            if not appid or appid in seen:
                continue
            seen.add(appid)
            title = row.find('span', class_='title').get_text(strip=True)
            summary = row.find('div', class_='search_review_summary')
            tooltip = summary.get('data-tooltip-html', '') if summary else ''
            m = re.search(r'([\d,]+)\s+reviews', tooltip)
            review_count = int(m.group(1).replace(',', '')) if m else 0
            games.append({'appid': appid, 'title': title, 'review_count': review_count})
            if len(games) >= n_games:
                break

        params['page'] += 1
        time.sleep(sleep_sec)

    return games

# 2) 한글 리뷰만, 길이 필터 적용해 크롤링
def crawl_reviews_for_app(app_id, max_per_app, writer, min_review_length,
                          language='korean', review_type='all',
                          purchase_type='all', sort_filter='recent',
                          num_per_page=100, sleep_sec=1.0):
    """
    app_id 하나에서 한국어 리뷰만, min_review_length 이상을 최대 max_per_app개 수집
    """
    url_tmpl = (
        'https://store.steampowered.com/appreviews/{app_id}'
        '?json=1&language={lang}'
        '&review_type={review_type}'
        '&purchase_type={purchase_type}'
        '&filter={sort_filter}'
        '&num_per_page={num_per_page}'
        '&cursor={cursor}'
    )
    accepted = 0
    cursor = '*'

    while accepted < max_per_app:
        url = url_tmpl.format(
            app_id=app_id,
            lang=urllib.parse.quote(language),
            review_type=review_type,
            purchase_type=purchase_type,
            sort_filter=sort_filter,
            num_per_page=num_per_page,
            cursor=urllib.parse.quote(cursor)
        )
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code != 200:
            break
        data = resp.json()
        reviews = data.get('reviews', [])
        if not reviews:
            break

        for r in reviews:
            text = r.get('review', '').strip().replace('\n', ' ')
            if len(text) < min_review_length:
                continue
            voted_up = int(r.get('voted_up', False))
            writer.writerow([app_id, voted_up, text])
            accepted += 1
            if accepted >= max_per_app:
                break

        cursor = data.get('cursor', '')
        time.sleep(sleep_sec)

    return accepted

# 3) 인기 게임 순으로 돌아가며 전체 리뷰 수집
def crawl_popular_games_reviews(output_csv,
                                total_max_reviews=1000,
                                per_app_limit=500,
                                top_n_games=50,
                                min_review_length=50):
    """
    - 대한민국 기준 Top Sellers 순(top_n_games) 게임들에서
    - 한글 리뷰만, 길이 ≥ min_review_length인 것들 중
    - 전체 total_max_reviews 만큼 수집
    - 게임당 per_app_limit 제한
    """
    games = get_top_games(top_n_games)
    collected_total = 0

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['appid', 'voted_up', 'review'])

        for game in games:
            if collected_total >= total_max_reviews:
                break
            appid = game['appid']
            remain = total_max_reviews - collected_total
            to_crawl = min(per_app_limit, remain)
            print(f"[{appid}] '{game['title']}' 에서 길이 ≥{min_review_length} 리뷰 {to_crawl}개까지 수집…")
            got = crawl_reviews_for_app(
                app_id=appid,
                max_per_app=to_crawl,
                writer=writer,
                min_review_length=min_review_length
            )
            collected_total += got
            print(f" → {got}개 수집 (누적 {collected_total}/{total_max_reviews})\n")

    print(f"완료! 총 {collected_total}개 리뷰를 '{output_csv}'에 저장했습니다.")


if __name__ == '__main__':
    # total = int(input("크롤링할 전체 리뷰 개수: "))
    # per_app = int(input("게임당 최대 리뷰 개수: "))
    # top_n = int(input("한국 기준 인기 게임 몇 개까지 순회? "))
    # min_len = int(input("최소 리뷰 길이(문자 수): "))
    crawl_popular_games_reviews(
        output_csv='kr_top_korean_reviews.csv',
        total_max_reviews=1000,
        per_app_limit=100,
        top_n_games=50,
        min_review_length=50
    )
