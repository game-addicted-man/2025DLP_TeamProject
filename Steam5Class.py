import pandas as pd
import numpy as np
from collections import Counter
from eunjeon import Mecab

# 1. 데이터 로딩 및 정제
df = pd.read_table('steam.txt', names=['label', 'reviews'])
df.drop_duplicates(subset=['reviews'], inplace=True)
df.dropna(how='any', inplace=True)

# 2. 형태소 분석 + 불용어 제거
mecab = Mecab()
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를',
             '인', '듯', '과', '와', '네', '들', '지', '임', '게', '만', '게임',
             '겜', '되', '음', '면', '에서', '니까', '어요', '니다']

df['reviews'] = df['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
df['tokenized'] = df['reviews'].apply(lambda x: [t for t in mecab.morphs(x) if t not in stopwords])

# 3. 긍정/부정 단어 Counter
positive_words = np.hstack(df[df['label'] == 1]['tokenized'].values)
negative_words = np.hstack(df[df['label'] == 0]['tokenized'].values)

# 1. 양쪽 word count
positive_word_count = Counter(positive_words)
negative_word_count = Counter(negative_words)

# 2. 모든 단어 집합
all_words = set(positive_word_count.keys()) | set(negative_word_count.keys())

# 3. 비율 기반 긍/부정 단어 선택
positive_vocab = set()
negative_vocab = set()
min_count = 3  # 너무 높으면 다 사라짐

for word in all_words:
    pos = positive_word_count[word]
    neg = negative_word_count[word]
    total = pos + neg
    if total < min_count:
        continue
    ratio = pos / total
    if ratio > 0.75:
        positive_vocab.add(word)
    elif ratio < 0.25:
        negative_vocab.add(word)


#print(" 부정 단어 TOP20:", negative_word_count.most_common(20))
#print(" 긍정 단어 TOP20:", positive_word_count.most_common(20))


# 5. 감성 점수 분류 함수 정의
def classify_sentiment(tokens, pos_words, neg_words):
    pos_count = sum(1 for word in tokens if word in pos_words)
    neg_count = sum(1 for word in tokens if word in neg_words)
    total = pos_count + neg_count

    if total == 0:
        return 2  # 중립

    ratio = pos_count / total

    # 더 완화된 기준
    if ratio >= 0.75:
        return 4  # 아주 긍정
    elif ratio >= 0.55:
        return 3  # 긍정
    elif ratio >= 0.45:
        return 2  # 중립
    elif ratio >= 0.25:
        return 1  # 부정
    else:
        return 0  # 아주 부정


# 6. 실제 감성 점수 적용
df['new_label'] = df['tokenized'].apply(lambda x: classify_sentiment(x, positive_vocab, negative_vocab))


for idx in range(10):
    tokens = df['tokenized'].iloc[idx]
    pos_count = sum(1 for word in tokens if word in positive_vocab)
    neg_count = sum(1 for word in tokens if word in negative_vocab)
    print(f"[{idx}] 리뷰: {df['reviews'].iloc[idx]}")
    print(f"  토큰: {tokens}")
    print(f"  긍정 단어 수: {pos_count}, 부정 단어 수: {neg_count}")
    print(f"  -> 감성 점수: {classify_sentiment(tokens, positive_vocab, negative_vocab)}\n")


# 7. 감성 점수 분포 확인
print("\n🎯 감성 점수 분포 (0~4):")
print(df['new_label'].value_counts().sort_index())

# 8. 저장
df[['new_label', 'reviews']].to_csv('steam_5class.txt', sep='\t', index=False, header=False)
print("✅ steam_5class.txt 파일 생성 완료!")