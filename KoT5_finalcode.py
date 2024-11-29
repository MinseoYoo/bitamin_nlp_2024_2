from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd
import numpy as np

# -----------------------------------모델 및 파일 불러오기--------------------------------------
tokenizer = AutoTokenizer.from_pretrained("Xcz2568/T5-summarization-Korean1")
model = AutoModelForSeq2SeqLM.from_pretrained("Xcz2568/T5-summarization-Korean1")

# special tokens 확인
# existing_special_tokens = tokenizer.special_tokens_map.get('additional_special_tokens', [])

# Load and preprocess the data
data = pd.read_csv('./train_개인및관계.csv')
data1 = pd.read_csv('./train_미용과건강.csv')
data2 = pd.read_csv('./train_상거래(쇼핑).csv')
data3 = pd.read_csv('./train_시사교육.csv')
data4 = pd.read_csv('./train_식음료.csv')
data5 = pd.read_csv('./train_여가생활.csv')
data6 = pd.read_csv('./train_일과직업.csv')
data7 = pd.read_csv('./train_주거와생활.csv')
data8 = pd.read_csv('./train_행사.csv')

train=pd.concat([data, data1, data2, data3, data4, data5, data6, data7, data8], ignore_index=True)

# Load and preprocess the data
data = pd.read_csv('./valid_개인및관계.csv')
data1 = pd.read_csv('./valid_미용과건강.csv')
data2 = pd.read_csv('./valid_상거래(쇼핑).csv')
data3 = pd.read_csv('./valid_시사교육.csv')
data4 = pd.read_csv('./valid_식음료.csv')
data5 = pd.read_csv('./valid_여가생활.csv')
data6 = pd.read_csv('./valid_일과직업.csv')
data7 = pd.read_csv('./valid_주거와생활.csv')
data8 = pd.read_csv('./valid_행사.csv')

valid=pd.concat([data, data1, data2, data3, data4, data5, data6, data7, data8], ignore_index=True)

# 역슬래시(\) 제거
valid['dialogue'] = valid['dialogue'].str.replace(r"\\", "", regex=True)

# 앰퍼샌드(&)가 포함된 행 삭제
valid = valid[~valid['dialogue'].str.contains(r"&", regex=True)]
valid = valid[~valid['dialogue'].str.contains(r"<", regex=True)]
valid = valid[~valid['dialogue'].str.contains(r">", regex=True)]

# 인덱스 재설정 (필요 시)
valid.reset_index(drop=True, inplace=True)

# 같은 발화자 연속 문장이면 발화자는 생략시키기
def merge_speaker_text(dialogue):
    dialogue_parts = dialogue.split()
    processed_dialogue = []
    current_speaker = dialogue_parts[0]
    processed_dialogue.append(current_speaker)
    
    for part in dialogue_parts:
        
        # 발화자 태그가 있는 경우
        if part.startswith("P") and part[-1] == ":":
            if current_speaker == part:
                continue
            else:
            # 다른 발화자의 경우 문장 추가 + 현재 발화자 바통터치
               processed_dialogue.append(part)
               current_speaker = part
               
        else:
            processed_dialogue.append(part)
            
    
    return " ".join(processed_dialogue)

train['processed_dialogue'] = train['dialogue'].apply(merge_speaker_text)
valid['processed_dialogue'] = valid['dialogue'].apply(merge_speaker_text)


def replace_speaker_tags(dialogue):
    dialogue = dialogue.replace("P01:", "<extra_id_0>")
    dialogue = dialogue.replace("P02:", "<extra_id_1>")
    dialogue = dialogue.replace("P03:", "<extra_id_2>")
    dialogue = dialogue.replace("P04:", "<extra_id_3>")
    dialogue = dialogue.replace("P05:", "<extra_id_4>")
    dialogue = dialogue.replace("P06:", "<extra_id_5>")
    dialogue = dialogue.replace("P07:", "<extra_id_6>")
    dialogue = dialogue.replace("P08:", "<extra_id_7>")
    dialogue = dialogue.replace("P09:", "<extra_id_8>")
    return dialogue

train['processed_dialogue'] = train['processed_dialogue'].apply(replace_speaker_tags)
valid['processed_dialogue'] = valid['processed_dialogue'].apply(replace_speaker_tags)

train.to_csv("train_all.csv", index=False, encoding='utf-8-sig')
valid.to_csv("valid_all.csv", index=False, encoding='utf-8-sig')

# -----------------------------------반복문자제거 및 맞춤법 교정-----------------------------------
# emoticon_normalize의 경우 멀쩡한 글자도 제거하는 경우도 존재 + 모델이 인식가능한 이모티콘이 존재해서 전처리를 하지않음
from soynlp.normalizer import emoticon_normalize, repeat_normalize 
# print(repeat_normalize('summarize: <extra_id_0> 자기 이제 슬슬 배고프겠네 <extra_id_1> 와 너 귀신이냐 <extra_id_0> ㅋㅋㅋㅋㅋㅋㅋㅋㅋ딱맞췄지 <extra_id_1> 너무배고파 어떡하지 <extra_id_0> ㅋㅋㅋㅋㅋㅋㅋㅋㅋ오늘은 진짜로 편의점 가서 뭐 사먹어라 이상한거 먹지말고!!!!', num_repeats=2))

from hanspell import spell_checker
#spell_checker.check("P01: 나는 내 나름대로 너네한테 노력한다고 비트윈도 하고 전화도 받은 건데 이렇게 너네 서운하게 할 거면 차라리 안 받고 나중에 바빴다고 말 할 걸 그랬나 봐 나는 그래도 평소에 잘한다고 생각했는데 한 번 못 하면 끝인 거 속상해 그것도 내 기준에서는 애인이랑 너네 둘 다 중요해서 노력한 거거든 받을 때도 내가 집중 잘 못 할 수 있는데 나갈까냐고 묻기도 했었고 근데 나 평소에 너네 제일 우선으로 두잖아 전화 거는 거나 받는 것도 그렇고 우리 전화할 때 나 안 받고 둘만 대화한 적 있긴 하냐... 나중에 온 전화들은 어차피 나 집중 못 한다고 그럴 것 같아서 안 받은 거야 잘 자라 P02: 장난이었는데 나 화 안 났어 그냥 네 반응 웃기고 귀여워서 계속 안 알려준다고 했던 거고 네가 잘 하는 거야 알지 아니까 그랬던 것 같아 안 잘 거 다 안다 자는 척 ㄴㄴ P01: #@이모티콘# 자는 건 아니지만 P01: 앤 진짜 바생이라 빨리 재우고 전화 집중할라구 그랬어... 미안 P02: 응 P02: 아냐 나도 미안 P02: #@이름#는 자냐 P01: 만날 삐치고 소리질러서 미안 그래도 친구해줘서 고마워 P01: 자나바 P01: 심심이두 미안")

train = pd.read_csv('train_all.csv')
valid = pd.read_csv('valid_all.csv')

error_rows = []  # 오류가 발생한 행의 인덱스 저장

def check_with_error_tracking(text, index):
    try:
        return spell_checker.check(text)[2]
    except Exception as e:
        error_rows.append(index)  # 오류 발생한 인덱스 저장
        print(f"Error at index {index}: {text}")
        print(f"Error message: {e}")
        return None  # 오류 발생 시 None 반환

# apply에 오류 핸들링 적용
train['processed_dialogue'] = train['processed_dialogue'].apply(
    lambda x: check_with_error_tracking(x, train.index[train['processed_dialogue'] == x].tolist()[0])
)

valid['processed_dialogue'] = valid['processed_dialogue'].apply(
    lambda x: check_with_error_tracking(x, valid.index[valid['processed_dialogue'] == x].tolist()[0])
)

# 오류가 발생한 행 출력
print("Rows with errors:", error_rows)

train.to_csv("train_all_preprocessed.csv", index=False, encoding='utf-8-sig')
valid.to_csv("valid_all_preprocessed.csv", index=False, encoding='utf-8-sig')

# 반복문자 전처리
train = pd.read_csv('train_all_preprocessed.csv')
valid = pd.read_csv('valid_all_preprocessed.csv')

import re

def custom_repeat_normalize_with_more_symbols(text, num_repeats=2):
    # 기본 반복 문자 정규화
    text = repeat_normalize(text, num_repeats=num_repeats)
    
    # 추가적인 특수문자 포함 처리
    text = re.sub(r"([!?.~]){2,}", lambda m: m.group(1) * num_repeats, text)
    
    return text


# 적용 전 전처리
data = valid #or train
nan_index=data.loc[data['processed_dialogue'].isna(),'processed_dialogue'].index
data.loc[data['processed_dialogue'].isna(),'processed_dialogue']= data.loc[data['processed_dialogue'].isna(),'dialogue']
data.loc[nan_index,'processed_dialogue'] = data.loc[nan_index,'processed_dialogue'].apply(merge_speaker_text)
data.loc[nan_index,'processed_dialogue'] = data.loc[nan_index,'processed_dialogue'].apply(replace_speaker_tags)

data['processed_dialogue']=data['processed_dialogue'].apply(lambda x: custom_repeat_normalize_with_more_symbols(x, num_repeats=2))

#data.to_csv("train_final_preprocessed.csv", index=False, encoding='utf-8-sig')
data.to_csv("valid_final_preprocessed.csv", index=False, encoding='utf-8-sig')


# ---------------------------시범예시---------------------------
# 모델 입력을 위해 "summarize:" 태그를 추가하여 텍스트 형식 지정, 앞에를 summarize:로 넣냐 summarization:으로 넣냐 아예 안넣냐에 따라서도 생성이 달라짐
# T5ForConditionalGeneration pre-trained 기본 형식 자체는 summarize: 였던 것 같음 

dialogue = """
<extra_id_0> 오늘 저녁에 뭐 먹을까?
<extra_id_1> 음... 나는 파스타가 먹고 싶은데 너는?
<extra_id_0> 파스타 좋지! 그런데 좀 새로운 메뉴가 없을까?
<extra_id_1> 그럼 이번에 새로 생긴 이탈리안 레스토랑 가볼래?
<extra_id_0> 음.. 모르겠다
<extra_id_1> 왜? 싫어?
"""

input_text = ["summarize: " + dialogue]
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(summary)

'''
reference : '지금 개봉한 영화 중에 뭘 본다고 했는지 이야기한다.'

기본상태
input = ['summarize: P01: #@이름# P01: 영화 P02: 웅? P01: 뭐본다헀어 P02: 엥 기생충? P01: 기생충말고 P02: 알라딘? P01: 지금개봉한거중에 P01: 없남']
output = 요약: 지금 개봉한 영화 중에 기생충 말고 알라딘이 있는지 물어본다.

발화자 이어 붙인 전처리 상태
input = ['summarize: P01: #@이름# 영화 P02: 웅? P01: 뭐본다헀어 P02: 엥 기생충? P01: 기생충말고 P02: 알라딘? P01: 지금개봉한거중에 없남']
output = 요약: 기생충 말고 알라딘이 개봉한 것 중에 없는지 묻고 있다.

발화자 이어 붙인 전처리 상태+ special token형식으로 변환된 상태
input = ['summarize: <extra_id_0> #@이름# 영화 <extra_id_1> 웅? <extra_id_0> 뭐본다헀어 <extra_id_1> 엥 기생충? <extra_id_0> 기생충말고 <extra_id_1> 알라딘? <extra_id_0> 지금개봉한거중에 없남']
output = 요약: 기생충 말고 알라딘을 보려고 하는데 지금 개봉한 것 중에 없는지 궁금하다.
'''

#-------------------Rouge 평가------------------- 
import evaluate
import pandas as pd

summary_df=pd.read_csv('summary_output_valid.csv') # 전처리 전 
#summary_df=pd.read_csv('summary_preprocessed_output.csv') # 전처리 후

# evaluate에서 ROUGE 메트릭 불러오기
rouge = evaluate.load("rouge")

# 참조 요약(reference summary)와 생성 요약(predicted summary) 정의
reference_summary = summary_df['reference']
predicted_summary = summary_df['summary']

# ROUGE 스코어 계산
results = rouge.compute(predictions=predicted_summary, references=reference_summary)

# 결과 출력
print("ROUGE-1:", results['rouge1']) # 전처리 전 : 0.08970805933867265 전처리 후 : 0.08968191832722547
print("ROUGE-2:", results['rouge2']) # 전처리 전 : 0.01825093005243844 전처리 후 : 0.01812076195605234
print("ROUGE-L:", results['rougeL']) # 전처리 전 : 0.08915009280324643 전처리 후 : 0.08917688031172902
print("ROUGE-Lsum:", results['rougeLsum']) # 전처리 전 : 0.0892209700874422 전처리 후 : 0.08916577813897537


#한글의 특성인 어근에 붙은 접사가( 은,는)가 단어의 역할을 결정하기에, 단어의 변형이 빈번하게 일어나므로 ROUGE 평가지표가 적절하지 않다는 판단에서 개발되었음
#--------------RDASS 평가-------------
import numpy as np
from transformers import AutoTokenizer
from scipy.spatial.distance import cosine
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 한국어 SBERT 모델 로드
sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# RDASS 계산 함수
def calculate_rdass(document, reference_summary, generated_summary):
    # 문서 및 요약문 벡터 추출
    v_d = sbert_model.encode(document)
    v_r = sbert_model.encode(reference_summary)
    v_p = sbert_model.encode(generated_summary)

        # s(p, r) 계산
    s_p_r = 1 - cosine(v_p, v_r)

    # s(p, d) 계산
    s_p_d = 1 - cosine(v_p, v_d)

    # RDASS 계산
    rdass = (s_p_r + s_p_d) / 2
    return rdass


def calculate_rdass_multiple(documents, reference_summaries, generated_summaries):
    rdass_scores = []
    for doc, ref, gen in tqdm(zip(documents, reference_summaries, generated_summaries)):
        # 단일 RDASS 계산
        rdass = calculate_rdass(doc, ref, gen)
        rdass_scores.append(rdass)
    return rdass_scores  # 여러 점수 반환

# RDASS 계산 (다수의 문장 처리)
documents = summary_df["processed_dialogue"].tolist()
reference_summaries = summary_df['reference'].tolist()
generated_summaries = summary_df['summary'].tolist()

# 다중 RDASS 계산
rdass_scores = calculate_rdass_multiple(documents, reference_summaries, generated_summaries)

# 평균 RDASS 점수 출력
average_rdass = np.mean(rdass_scores)
print(f"Average RDASS Score: {average_rdass}") # 전처리 전 RDASS score = 0.5707704020678929, 전처리 후 RDASS Score: 0.5795888804860062

# kobart
kobart=pd.read_csv('dev_v1_KOBART.csv')

kobart['abstractive'][0:10]
#--------------------------------------------- 파인튜닝 시작-----------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Xcz2568/T5-summarization-Korean1")
model = AutoModelForSeq2SeqLM.from_pretrained("Xcz2568/T5-summarization-Korean1")

train=pd.read_csv('train_all.csv')

input_texts = ["summarize: " + text for text in train['processed_dialogue']]
target_texts = train['summary'].tolist()

# 입력과 출력 텍스트를 토큰화
inputs = tokenizer(
    input_texts,
    max_length=1580,  # model max_length는 128이지만 input 문장길이에 맞춰 설정하니 요약생성이 더 잘됐음
    padding=True,
    truncation=True,
    return_tensors="pt",
    
)

labels = tokenizer(
    target_texts,
    max_length=128, # train['summary']의 길이는 123이지만 효율적 계산 처리를 위해 128로 맞춤  
    padding=True,
    truncation=True,
    return_tensors="pt"
)

import torch

class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels["input_ids"][idx]
        }

# Dataset 생성
dataset = T5Dataset(inputs, labels) # shape = torch.Size([279992, 1580])

'''
# `DataParallel`을 사용하여 Multi-GPU 활성화
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.parallel.DataParallel(model)
'''
from transformers import AutoModelForSeq2SeqLM, AdamW

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

from torch.utils.data import DataLoader

# DataLoader 생성
train_loader = DataLoader(dataset, batch_size=12, shuffle=True) 

from tqdm import tqdm

# 모델 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 루프
epochs = 5  
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # 데이터 이동
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 모델 출력 계산
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 진행 상태 출력
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
        
model.save_pretrained("./fine_tuned_t5")
tokenizer.save_pretrained("./fine_tuned_t5")

#=-------------------------- train 요약생성 최종(전처리 전) ----------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import torch  # GPU 사용을 위한 PyTorch 모듈

# 모델 및 데이터 로드
tokenizer = AutoTokenizer.from_pretrained("Xcz2568/T5-summarization-Korean1") # or "./fine_tuned_t5"
model = AutoModelForSeq2SeqLM.from_pretrained("Xcz2568/T5-summarization-Korean1") # or "./fine_tuned_t5"

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train = pd.read_csv('train_all.csv')
# 입력 텍스트 생성
input_text = ["summarize: " + text for text in train['processed_dialogue']]

# 요약 생성
summary_data = []
batch_size = 12

for start_idx in tqdm(range(0, len(input_text), batch_size)):
    batch_texts = input_text[start_idx:start_idx+batch_size]
    
    # 입력 데이터 토큰화
    inputs = tokenizer(batch_texts, return_tensors="pt",padding=True,truncation=True,max_length=1580)#1580
    token_lengths = [len(input_id) for input_id in inputs["input_ids"]]
    # print(f"각 텍스트의 토큰화된 길이: {token_lengths}")
    # print(inputs)
    # 입력 데이터를 GPU로 이동
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        # 모델에서 요약 생성
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=150, 
            num_beams=5, 
            early_stopping=True
        )
        
        # 결과 디코딩
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summary_data.extend(summaries)
    except Exception as e:
        print(f"Error processing batch {start_idx}: {e}")
        summary_data.extend([""] * len(batch_texts))

# 결과 저장
summary_df = pd.DataFrame({
    "processed_dialogue": train['processed_dialogue'],
    'reference': train['summary'],
    "summary": summary_data
})

summary_df.to_csv("summary_output_train.csv", index=False, encoding='utf-8-sig')



#=-------------------------------- valid 요약생성 최종(전처리 전) --------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import torch 

# 모델 및 데이터 로드
tokenizer = AutoTokenizer.from_pretrained("Xcz2568/T5-summarization-Korean1") # or "./fine_tuned_t5"
model = AutoModelForSeq2SeqLM.from_pretrained("Xcz2568/T5-summarization-Korean1") # or "./fine_tuned_t5"

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

valid = pd.read_csv('valid_all.csv')

# 입력 텍스트 생성
input_text = ["summarize: " + text for text in valid['processed_dialogue']]

# 요약 생성
summary_data = []
batch_size = 11

for start_idx in tqdm(range(0, len(input_text), batch_size)):
    batch_texts = input_text[start_idx:start_idx+batch_size]
    
    # 입력 데이터 토큰화
    inputs = tokenizer(batch_texts, return_tensors="pt",padding=True,truncation=True,max_length=850) # valid max는 850
    #token_lengths = [len(input_id) for input_id in inputs["input_ids"]]
    #print(f"각 텍스트의 토큰화된 길이: {token_lengths}")

    # 입력 데이터를 GPU로 이동
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        # 모델에서 요약 생성
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=150, 
            num_beams=5, 
            early_stopping=True
        )
        
        # 결과 디코딩
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summary_data.extend(summaries)
    except Exception as e:
        print(f"Error processing batch {start_idx}: {e}")
        summary_data.extend([""] * len(batch_texts))

# 결과 저장
summary_df = pd.DataFrame({
    "processed_dialogue": valid['processed_dialogue'],
    'reference': valid['summary'],
    "summary": summary_data
})

summary_df.to_csv("summary_output.csv", index=False, encoding='utf-8-sig')

#=-------------------------------- valid 요약생성 최종(전처리 후) --------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import torch  

# 모델 및 데이터 로드
tokenizer = AutoTokenizer.from_pretrained("Xcz2568/T5-summarization-Korean1") # or "./fine_tuned_t5"
model = AutoModelForSeq2SeqLM.from_pretrained("Xcz2568/T5-summarization-Korean1") # or "./fine_tuned_t5"

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

valid = pd.read_csv('valid_final_preprocessed.csv')

# 입력 텍스트 생성
input_text = ["summarize: " + text for text in valid['processed_dialogue']]

# 요약 생성
summary_data = []
batch_size = 11


for start_idx in tqdm(range(0, len(input_text), batch_size)):
    batch_texts = input_text[start_idx:start_idx+batch_size]
    
    # 입력 데이터 토큰화
    inputs = tokenizer(batch_texts, return_tensors="pt",padding=True,truncation=True,max_length=850) 
    #token_lengths = [len(input_id) for input_id in inputs["input_ids"]]
    #print(f"각 텍스트의 토큰화된 길이: {token_lengths}")

    # 입력 데이터를 GPU로 이동
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        # 모델에서 요약 생성
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=150, 
            num_beams=5, 
            early_stopping=True
        )
        
        # 결과 디코딩
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summary_data.extend(summaries)
    except Exception as e:
        print(f"Error processing batch {start_idx}: {e}")
        summary_data.extend([""] * len(batch_texts))

# 결과 저장
summary_df = pd.DataFrame({
    "processed_dialogue": valid['processed_dialogue'],
    'reference': valid['summary'],
    "summary": summary_data
})

summary_df.to_csv("summary_preprocessed_output.csv", index=False, encoding='utf-8-sig')