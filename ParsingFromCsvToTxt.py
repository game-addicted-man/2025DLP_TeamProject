import csv
import re

def is_korean(text):
    #한글이 1글자 이상 포함되어 있으면 True
    return bool(re.search(r'[가-힣]', text))

input_csv = 'kr_top_korean_reviews.csv'
output_txt = 'kr_reviews_labeled.txt'

with open(input_csv, 'r', encoding='utf-8-sig') as f_in, \
     open(output_txt, 'w', encoding='utf-8') as f_out:
    
    reader = csv.DictReader(f_in)
    for row in reader:
        label = row['voted_up']
        review = row['review'].strip().replace('\n', ' ')
        
        # 한글 리뷰만 필터링
        if not is_korean(review):
            continue

        # 라벨 \t 텍스트 형태로 저장
        f_out.write(f"{label}\t{review}\n")