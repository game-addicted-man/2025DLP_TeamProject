import pandas as pd
import numpy as np
import re
from eunjeon import Mecab
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터 불러오기
df = pd.read_table('steam_5class.txt', names=['label', 'reviews'])

# 2. 전처리 및 토큰화
mecab = Mecab()
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를',
             '인', '듯', '과', '와', '네', '들', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']
df['reviews'] = df['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
df.dropna(how='any', inplace=True)
df['tokenized'] = df['reviews'].apply(mecab.morphs)
df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stopwords])

# 3. 훈련/테스트 분리
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)

# 4. 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['tokenized'])

# 희귀 단어 제외
threshold = 2
total_cnt = len(tokenizer.word_index)
rare_cnt = sum(1 for _, v in tokenizer.word_counts.items() if v < threshold)
vocab_size = total_cnt - rare_cnt + 2

tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(train_data['tokenized'])

X_train = tokenizer.texts_to_sequences(train_data['tokenized'])
X_test = tokenizer.texts_to_sequences(test_data['tokenized'])

max_len = max(len(x) for x in X_train)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 5. 레이블 원-핫 인코딩
y_train = to_categorical(train_data['label'].values, num_classes=5)
y_test = to_categorical(test_data['label'].values, num_classes=5)

# 6. 모델 정의
embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units)))
model.add(Dense(5, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
mc = ModelCheckpoint('best_model_5class.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=256, validation_split=0.2)

# 7. 예측 함수
def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = mecab.morphs(new_sentence)
    new_sentence = [word for word in new_sentence if word not in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)
    pred = model.predict(pad_new)
    score = np.argmax(pred)
    labels = ['아주 부정적', '부정적', '중립', '긍정적', '아주 긍정적']
    print("감성 예측 결과:", labels[score])
    print("확률 분포:", pred[0])

# 8. 테스트
sentiment_predict('개씨발 좆같은 게임')
sentiment_predict('진짜 ㄹㅇ 초갓겜, 게임성부터 노래까지 미쳤다 그냥')
sentiment_predict('ㄹㅇ 병신겜같이 보이는데 초갓겜임, 처음엔 좀 그런데 하다보면 중독성 개쩜, 개재밌음 ㄹㅇㅇ')
sentiment_predict('노래 좋고, 게임성은 평균, 그렇게까지 재밌지는 않았지만.. 그래도 수작')
sentiment_predict('애매하다 애매해.. 그렇다고 개똥겜까지는 아님 ㅋㅋ')