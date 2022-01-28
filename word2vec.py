import nltk
nltk.download('punkt')

from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

import itertools
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, WordPunctTokenizer, TreebankWordTokenizer, RegexpTokenizer, sent_tokenize
from multilabel_pipeline import MultiLabelPipeline
from transformers import ElectraTokenizer
from model import ElectraForMultiLabelClassification
import preprocess_text
import re
from konlpy.tag import *
okt = Okt()

tokenizer_goemotions = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-goemotions")
model_goemotions = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-v3-goemotions")

goemotions = MultiLabelPipeline(
    model=model_goemotions,
    tokenizer=tokenizer_goemotions,
    threshold=0.3
)

# tokens 경로에서 리스트 데이터 불러오기
tokens = preprocess_text.tokens
tokens = list(itertools.chain.from_iterable(tokens))


# 진짜 토크나이징한 리스트 real_tokens
real_tokens = []
for i in range(len(tokens)):
    real_tokens.append(word_tokenize(tokens[i]))


# word2vec 모델 불러와서 학습하기
model = Word2Vec(real_tokens, alpha=0.025, window=3, min_count=5, sg=1)
model.train(real_tokens, total_examples=len(real_tokens), epochs=30)


# soynlp 사용을 위한 txt 파일 만들어주기 (기존 파일 기준)

# with open('data.txt', 'w') as f:
#     for line in tokens:
#         f.write(line)


# [응용] 명사 추출한 것 바탕으로 유사어 검색 자동화하기

# 문장에 포함된 명사를 알려주는 함수
def show_nouns(text):
    for i in range(len(text)):
        noun_lst = okt.nouns(text[i])
    # for j in range(len(noun_lst)):
    #     if len(noun_lst[j]) == 1:
    #         noun_lst.remove(noun_lst[j])
    return noun_lst


# 문장에 포함된 동사를 알려주는 함수
def show_verbs(text):
    okts = []
    verbs = []
    for i in range(len(text)):
        okts.append(okt.pos(text[i]))
    for j in range(len(okts)):
        for k in range(len(okts[j])):
            if okts[j][k][1] == 'Verb' and len(okts[j][k][0]) > 1:
                verbs.append(okts[j][k][0])
    return verbs


# 문장에 포함된 형용사를 알려주는 함수
def show_adjectives(text):
    okts = []
    adjs = []
    for i in range(len(text)):
        okts.append(okt.pos(text[i]))
    for j in range(len(okts)):
        for k in range(len(okts[j])):
            if okts[j][k][1] == 'Adjective' and len(okts[j][k][0]) > 1:
                adjs.append(okts[j][k][0])
    return adjs


# 단어와 연관된 유사어를 알려주는 함수
def find_similar_words(text):
    similar_words_lst = []
    for i in range(len(text)):
        try:
            similar_words_lst.append({text[i]: model.wv.most_similar(text[i])})
        except:
            pass

    return similar_words_lst

# 유사어 자체에서 품사별 추출해서 알려주는 함수

# 명사
def extract_from_similar_nouns(lst):
    real_similar_words_lst = []
    nouns_lst = []
    for i in range(len(lst)):
        value = list(itertools.chain.from_iterable(list(lst[i].values())))
        real_similar_words_lst.extend(re.compile('[가-힣]+').findall(str(value)))

    for j in range(len(real_similar_words_lst)):
        nouns_lst.append(show_nouns([real_similar_words_lst[j]]))
    real_similar_words_lst = set(list(itertools.chain.from_iterable(list(nouns_lst)))) # 중복 제거
    return real_similar_words_lst


# 문장이 내포하는 감정을 알려주는 함수
def show_emotion(text):
    for i in range(len(text)):
        emotion_lst = goemotions(text[i])
    return emotion_lst


# 문장에 들어있는 명사/동사/형용사 추출 + 감정 분류 함수 실행 코드
def show_emotion_with_text(text):
    print('target text:', text)
    print('emotion label:', show_emotion(text))
    print('\n')
    print('1st noun words:', show_nouns(text))
    print('2nd noun words:', extract_from_similar_nouns(find_similar_words(show_nouns(text))))
    print('\n')
    # print('verb words:', show_verbs(text))
    # print('similar words w/ verb:', find_similar_words(show_verbs(text)))
    # print('\n')
    # print('adjective words:', show_adjectives(text))
    # print('similar words w/ adj:', find_similar_words(show_adjectives(text)))
    # print('\n')
    return


# 예시로 데이터에는 없는 raw text 를 넣어보자
# 메타포
data = ['봄은 고양이로다']
show_emotion_with_text(data)

# 랩 가사
data = ['좋은 바이브 좋은 밤 꿈을 꽤 비싼 값에 샀어']
show_emotion_with_text(data)

# 랩 가사
data = ['중심을 잃고 목소리도 잃고 비난받고 사람들과 멀어지는 착각 속에']
show_emotion_with_text(data)

# kpop 가사
data = ['어차피 내가 살아 내 인생 내 거니까']
show_emotion_with_text(data)

# kpop 가사
data = ['다시 너와 연결될 수 있다면 너를 만나고 싶어 이제']
show_emotion_with_text(data)

# kpop 가사
data = ['절대적 룰을 지켜 결속은 나의 무기']
show_emotion_with_text(data)

# 시 구절
data = ['날개야 다시 돋아라. 날자. 날자. 날자. 한 번만 더 날자꾸나']
show_emotion_with_text(data)

# 시 구절
data = ['받아들이면 된다 지는 해를 깨우려 노력하지 말거라 너는 달빛에 더 아름답다']
show_emotion_with_text(data)

# 에피소드
data = ['밤늦게까지 일하고 집에 돌아와 잠을 못 이기고 쓰러진 어머니']
show_emotion_with_text(data)

# 에피소드
data = ['귀여운 고양이를 쓰다듬었다']
show_emotion_with_text(data)

# 에피소드
data = ['커피 향기가 너무 좋다']
show_emotion_with_text(data)



