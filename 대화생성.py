pip install u sentence_transformers
from sentence_transformers import SentenceTransformer, util

# KoSentenceBERT 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')


# 대화 내용과 질문 리스트
dialogue = 'P01: 선생님 P01: 혹쉬 바지 어디서 구입하시나요 P02: 저 육육 거기서 구입해유 거기가 바지는 괜찮아유 P01: 너 거기서 바지사면 m입나여? P02: ㅇㅇ m 근데 후기 잘보고 P02: 바지마다 사이즈가 다른게 잇거든 P01: 사람들후기에 P01: 쇼핑몰 후기 ! P01: 거기 몸무게 믿어도되나 P02: 후기 여러개 바야댐,,, 그래서 공통적이거만 보구,,, 옷 상세치수도 보고 구래,,,'
summary = '육육이 바지가 괜찮지만 사이즈 등은 후기를 잘 보고 사야 한다.'
questions = ["어떤 스타일의 옷 좋아해?", "보통 사이즈 몇 입어?", "무슨 바지를 주로 입어?", '좋아하는 가수는?', '무슨 노래가 괜찮아??']

# 임베딩 생성
dialogue_embedding = model.encode(summary, convert_to_tensor=True)
question_embeddings = model.encode(questions, convert_to_tensor=True)

# 유사도 계산
similarities = util.cos_sim(dialogue_embedding, question_embeddings)
similarities # tensor([[0.4363, 0.5648, 0.5539, 0.1244, 0.0833]])

# 가장 높은 유사도를 가진 질문 선택
best_question_index = similarities.argmax()
best_question = questions[best_question_index] # '보통 사이즈 몇 입어?'
