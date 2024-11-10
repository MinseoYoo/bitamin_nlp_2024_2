import json
import pandas as pd

# JSON 파일 로드
with open('./sample.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 대화 데이터를 추출하여 리스트로 변환
dialogue_data = []
for item in data['data']:
    fname = item['header']['dialogueInfo']['dialogueID']
    
    # 화자 정보를 포함한 대화 내용 생성
    dialogue = ' '.join([f"{utterance['participantID']}: {utterance['utterance']}" for utterance in item['body']['dialogue']])
    
    summary = item['body']['summary']
    dialogue_data.append({'fname': fname, 'dialogue': dialogue, 'summary': summary})

# DataFrame으로 변환
df = pd.DataFrame(dialogue_data)

# CSV 파일로 저장
df.to_csv('./train.csv', index=False, encoding='utf-8-sig')



# JSON 파일 로드
with open('./개인및관계.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 대화 데이터를 추출하여 리스트로 변환
dialogue_data = []
for item in data['data']:
    fname = item['header']['dialogueInfo']['dialogueID']
    
    # 화자 정보를 포함한 대화 내용 생성
    dialogue = ' '.join([f"{utterance['participantID']}: {utterance['utterance']}" for utterance in item['body']['dialogue']])
    
    summary = item['body']['summary']
    dialogue_data.append({'fname': fname, 'dialogue': dialogue, 'summary': summary})

# DataFrame으로 변환
df = pd.DataFrame(dialogue_data)

# CSV 파일로 저장
df.to_csv('./dev.csv', index=False, encoding='utf-8-sig')

