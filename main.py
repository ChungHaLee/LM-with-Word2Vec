from multilabel_pipeline import MultiLabelPipeline
from transformers import ElectraTokenizer
from model import ElectraForMultiLabelClassification
from pprint import pprint
from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2
from konlpy.tag import *


okt = Okt()

tokenizer_goemotions = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-goemotions")
model_goemotions = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-v3-goemotions")

goemotions = MultiLabelPipeline(
    model=model_goemotions,
    tokenizer=tokenizer_goemotions,
    threshold=0.3
)

# 데이터가 적을 때에는 soynlp 보다 konlpy 사용
# 텍스트 input
data = [
 "한 여자가 알프스 산맥에 가서 나비를 만난다.",
 "전화를 통해 안좋은 소식을 접한 것 같다.",
 "분위기 좋은 카페에 가서 마시는 한 잔의 커피",
 "이번에 데뷔한 여자 아이돌 그룹 예쁘더라"
]

def show_noun_with_emo(text):
    for i in range(len(text)):
        print('명사 추출:', okt.nouns(text[i]))
        print('감정 예측:', goemotions(text[i]))
        print('\n')
    return

# 문장과 관련된 명사를 알려주는 함수
def show_nouns(text):
    for i in range(len(text)):
        noun_lst = okt.nouns(text[i])
    return noun_lst

# 문장이 내포하는 감정을 알려주는 함수
def show_emotion(text):
    for i in range(len(text)):
        emotion_lst = goemotions(text[i])
    return emotion_lst

show_noun_with_emo(data)