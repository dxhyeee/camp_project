from mpi4py import MPI
import pandas as pd
import re

# 1. MPI 설정
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ==========================================
# 🔍 [사용자 시나리오 입력]
# 사용자가 자신의 상황을 구체적으로 적습니다.
# ==========================================
user_input = """
저는 강남구에서 친구에게 투자를 권유받았습니다. 
높은 이자를 준다고 해서 3천만 원을 보냈는데, 
알고 보니 다 거짓말이었고 돈을 돌려주지 않고 있습니다. 
이거 사기죄 성립되나요?
"""

# 2. NLP 전처리 함수 (AI의 '학습/분석' 로직)
def analyze_text(text):
    # (1) 의미 없는 조사/어미 제거 (노이즈 필터링)
    stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', 
                 '합니다', '습니다', '했다', '이다', '하고', '하여', '된', '인', '저', '제']
    
    # (2) 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # (3) 핵심 키워드 추출 (2글자 이상 명사 추정 단어)
    keywords = set()
    for word in words:
        clean_word = word
        for stop in stopwords:
            if clean_word.endswith(stop):
                clean_word = clean_word[:-len(stop)]
        if len(clean_word) >= 2:
            keywords.add(clean_word)
            
    return keywords

# 3. 유사도 분석 (분석 로직)
def get_match_score(query_keywords, case_keywords):
    # 교집합: 사용자와 판례 간 공통된 핵심 단어
    intersection = query_keywords.intersection(case_keywords)
    # 합집합: 전체 단어 풀
    union = query_keywords.union(case_keywords)
    
    if not union: return 0.0, set()
    
    score = len(intersection) / len(union)
    return score, intersection  # 점수와 '매칭된 단어들'을 함께 반환

# ==========================================
# 메인 실행 로직
# ==========================================

# Rank 0: 데이터 로드 및 분배
if rank == 0:
    try:
        df = pd.read_csv('legal_data_perfect.csv') # 최종 데이터 파일
        all_data = df.to_dict('records')
        
        # 데이터 분할 (Chunking)
        chunk_size = len(all_data) // size
        chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
        if len(chunks) > size: chunks[-1].extend(chunks[size:]); chunks = chunks[:size]
            
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        comm.Abort()
else:
    chunks = None

my_data = comm.scatter(chunks, root=0)

# 각 프로세스별 분석 수행
my_results = []
query_keywords = analyze_text(user_input) # 사용자 입력 분석

for case in my_data:
    # 판례 분석 (키워드 추출)
    case_keywords = analyze_text(case['Facts'])
    
    # 유사도 및 매칭 근거 산출
    score, matched_words = get_match_score(query_keywords, case_keywords)
    
    my_results.append({
        'Case_ID': case['Case_ID'],
        'Category': case['Category'],
        'Score': score,
        'Matched': matched_words, # 분석의 근거 (매칭된 단어)
        'Facts': case['Facts']
    })

# 각자 찾은 TOP 3 추출
my_results = sorted(my_results, key=lambda x: x['Score'], reverse=True)[:3]

# 결과 취합
gathered_results = comm.gather(my_results, root=0)

# Rank 0: 최종 분석 리포트 출력
if rank == 0:
    final_candidates = [item for sublist in gathered_results for item in sublist]
    final_top3 = sorted(final_candidates, key=lambda x: x['Score'], reverse=True)[:3]
    
    print("\n" + "="*60)
    print(f"🕵️  [AI 법률 분석 리포트]")
    print("="*60)
    print(f"📝 사용자 입력 요약: {user_input.strip()[:50]}...")
    print(f"🔑 사용자 핵심 키워드: {query_keywords}")
    print("-" * 60)
    
    for i, res in enumerate(final_top3):
        print(f"🏆 추천 판례 {i+1}위: [{res['Category']}] (적합도: {res['Score']:.4f})")
        print(f"   ID: {res['Case_ID']}")
        print(f"   💡 분석 결과 (매칭된 핵심 정황): {res['Matched']}")
        print(f"   📜 판례 내용: {res['Facts'][:100]}...")
        print("-" * 60)
