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
 "배경이 겨울의 도시이고 파란색을 중심색으로 사용하여 인물의 쓸쓸한 감정이 잘 드러난다.",
 "따뜻하고 기분이 좋아지는 느낌이다.",
 "우울하고 외로운 느낌.",
 "남자친구가 말도안되는 변명을 해서 대답할가치도 못 느끼는 모습.",
 "전교 1등을 미워하는 전교 2등의 살인계획",
 "아주 아름다운 풍경을 지닌 장소를 태연이 만끽하고 있다.",
 "초록 풀잎들이 잔뜩 보이는 한적한 시골 마을",
 "자존감 낮은 부모가 아이를 학대하며 스스로를 위로하는 장면",
 "영화에서 주인공이 들키면 안될 비밀이 발각된 느낌의 장면",
 "회사에서 실수한 1개월차 신입사원.",
 "아직 눈도 마주치지 못 하는 새로운 커플의 모습",
 "소개팅 나온 남자의 어이없는 시리얼 먹방",
 "긴장되는 창고에서 조직을 만나는 장면",
 "아이의 행동을 용납할 수 없었던 엄마, 아이를 던지다.",
 "누군가를 죽이고 싶을 때, 그러지 못 할 때.",
 "우는 모습도 예쁜 여배우들의 연기",
 "당당한 언니들의 패션쇼",
 "사이비 신도들의 소름돋는 모습들",
 "엄마와의 추억",
 "학창시절 친구들과 함께했던 추억들.",
 "게임 할 때 듣는 음악 배경으로 깔리는 네온 영상.",
 "오케스트라나 밴드처럼 다양한 악기를 사용한 음악이 떠오른다.",
 "메두사에게 최면이 걸려 감급된 공주의 모습이 떠오른다.",
 "불사조가 날아가는 모습이 연상되었다.",
 "헤엄쳐오는 해파리같기도 하고, 셔틀콕같다는 생각도 들었다.",
 "어두운 암실에서 레이저가 쏘아지는 것 같은 느낌이 들었다.",
 "시간이동이나 장소이동을 할 때 나올 것 같은 장면같다고 생각했다.",
 "요정이나 마법사들같다는 생각이 들었다.",
 "비가 오자 비가 오는 걸 좋아했던 연인이 떠오른 것 같다.",
 "연인과 과거에 같이 여행왔던 곳을 혼자 쓸쓸히 돌아다니고 있다는 생각이 든다.",
 "감기에 걸린 남자가 나른하고 어지러운 느낌에 대해 말하고 있을 것 같다.",
 "수현은 썸을 타는 남자와 문자를 하고 있었을 것 같다.",
 "능력을 숨긴 채 착하게만 살던 주인공이 어떤 일을 계기로 바뀌려고 다짐하는 이야기가 떠오른다.",
 "마을에서 도망쳐 자유를 만끽하는 중세시대 소녀의 모습이 떠올랐다.",
 "꽃의 정령이 깨어난 것 같다.",
 "이상한 나라의 엘리스가 떠오른다.",
 "단순한 슬픔을 넘어서 오열하는 사람들의 모습이 떠오른다.",
 "나를 반갑게 맞아주시는 할머니가 떠올랐다.",
 "트와이스가 추는 TT 춤이 생각난다."
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

# show_noun_with_emo(data)