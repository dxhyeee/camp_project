from mpi4py import MPI
import pandas as pd
import numpy as np
import re
from collections import Counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. 자연어 처리: 단어를 특징(Feature)으로 추출
def get_vector(text):
    stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '합니다', '습니다']
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    # 조사 제거 및 빈도 계산
    clean_words = [w for w in words if w not in stopwords and len(w) > 1]
    return Counter(clean_words)

# 2. 코사인 유사도 계산: 두 벡터 사이의 '의미적 거리(각도)' 측정
# 이것이 단어 의미 학습(Embedding)의 핵심 수학 원리입니다.
def cosine_similarity(v1, v2):
    # 공통 단어 추출
    common = set(v1.keys()) & set(v2.keys())
    if not common: return 0.0
    
    # 분자: 두 벡터의 내적 (Dot Product)
    dot_product = sum(v1[x] * v2[x] for x in common)
    
    # 분모: 두 벡터의 크기 (Magnitude)의 곱
    norm1 = np.sqrt(sum(v1[x]**2 for x in v1.keys()))
    norm2 = np.sqrt(sum(v2[x]**2 for x in v2.keys()))
    
    return dot_product / (norm1 * norm2)

# ==========================================
# 실행 로직 (사용자 입력 분석)
# ==========================================
user_input = "강남구에서 고수익 투자를 보장한다며 3천만 원을 빌려간 뒤 잠적한 사기꾼을 신고하고 싶습니다."
query_vector = get_vector(user_input)

# Rank 0: 데이터 로드
if rank == 0:
    df = pd.read_csv('legal_data_perfect.csv')
    all_data = df.to_dict('records')
    chunk_size = len(all_data) // size
    chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
else:
    chunks = None

my_data = comm.scatter(chunks, root=0)
my_results = []

for case in my_data:
    case_vector = get_vector(case['Facts'])
    # 단순 키워드 매칭이 아닌 '벡터 공간 상의 각도' 분석
    score = cosine_similarity(query_vector, case_vector)
    my_results.append({'Case_ID': case['Case_ID'], 'Category': case['Category'], 'Score': score, 'Facts': case['Facts']})

my_results = sorted(my_results, key=lambda x: x['Score'], reverse=True)[:3]
gathered_results = comm.gather(my_results, root=0)

if rank == 0:
    final_candidates = [item for sublist in gathered_results for item in sublist]
    final_top3 = sorted(final_candidates, key=lambda x: x['Score'], reverse=True)[:3]
    
    print("\n" + "="*60)
    print("🧠 [AI 벡터 공간 모델 기반 판례 분석 결과]")
    print("="*60)
    for i, res in enumerate(final_top3):
        print(f"Rank {i+1}: [{res['Category']}] 유사도: {res['Score']:.4f}")
        print(f"ID: {res['Case_ID']} / 요약: {res['Facts'][:80]}...")
    print("="*60)
