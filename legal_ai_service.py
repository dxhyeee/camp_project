from mpi4py import MPI
import pandas as pd
import numpy as np
import re

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# ==================================================================
# 📝 [사용자 입력] 여기에 판례를 찾고 싶은 사연을 적으세요!
# ==================================================================
user_input = """
강남구 역삼동에서 친구가 고수익을 보장한다며 5천만 원을 빌려갔는데,
알고 보니 도박에 다 탕진하고 연락을 끊고 잠적했습니다. 
이 사람을 처벌할 수 있을까요?
"""
# ==================================================================

# '하드 모드' 전처리 함수 (아까 학습된 그 로직 그대로 사용)
# 정답 단어(사기, 절도 등) 없이 오직 '정황'만으로 매칭함
def get_inference_vector(text):
    cheat_words = ['사기', '절도', '마약', '횡령', '폭행', '음주운전', '명예훼손', '교통사고', 
                   '공무집행방해', '강제추행', '사건', '혐의', '피고인', '판결', '징역', '무죄', 
                   '선고', '기소', '재판부', '상당', '피해', '발생']
    text = re.sub(r'[^\w\s]', '', text)
    # 치트 단어 제외하고 문맥 단어만 추출
    words = [w for w in text.split() if w not in cheat_words and len(w) > 1]
    return set(words)

# 1. 데이터 로드 (전체 1,600개 데이터 사용)
if rank == 0:
    try:
        df = pd.read_csv('legal_data_total.csv')
        all_cases = df.to_dict('records')
    except:
        print("❌ 데이터 파일이 없습니다.")
        comm.Abort()
else:
    all_cases = None

# 데이터 분산 (12개 코어가 1,600개를 나눠서 검색)
if rank == 0:
    chunks = np.array_split(all_cases, size)
else:
    chunks = None

my_chunk = comm.scatter(chunks, root=0)

# 2. 병렬 검색 (Similarity Search)
my_results = []
user_vec = get_inference_vector(user_input)

for case in my_chunk:
    case_vec = get_inference_vector(case['Facts'])
    
    # 유사도 계산 (Jaccard Similarity)
    if not (user_vec | case_vec): score = 0
    else: score = len(user_vec & case_vec) / len(user_vec | case_vec)
    
    # 점수가 0점보다 높은 경우만 후보로 등록
    if score > 0:
        my_results.append({
            'Category': case['Category'],
            'Score': score,
            'Facts': case['Facts'],
            'Match_Keywords': list(user_vec & case_vec) # 매칭된 단어 추적
        })

# 3. 결과 취합 (Gather)
gathered_results = comm.gather(my_results, root=0)

# 4. Rank 0가 최종 리포트 출력
if rank == 0:
    # 모든 결과 합치기
    final_candidates = [item for sublist in gathered_results for item in sublist]
    # 점수 높은 순으로 정렬해서 TOP 3 뽑기
    top3 = sorted(final_candidates, key=lambda x: x['Score'], reverse=True)[:3]
    
    print("\n" + "="*70)
    print(f"🤖 [HPC AI 법률 상담 리포트] (Processors: {size})")
    print("="*70)
    print(f"📌 사용자 사연 요약: {user_input.strip()[:50]}...")
    print("-" * 70)
    
    if not top3:
        print("죄송합니다. 유사한 판례를 찾지 못했습니다. 내용을 더 구체적으로 적어주세요.")
    else:
        for i, res in enumerate(top3):
            print(f"🏆 추천 판례 {i+1}위")
            print(f"   📂 죄명 분류: [{res['Category']}]")
            print(f"   📊 유사도 점수: {res['Score']*100:.2f}%")
            print(f"   🔑 매칭된 정황: {res['Match_Keywords']}")
            print(f"   📜 판례 내용: {res['Facts'][:100]}...")
            print("-" * 70)
