from mpi4py import MPI
import pandas as pd
import numpy as np
import re

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# 1. 사용자 요청 단계 (점점 많은 데이터를 보여줌)
learning_phases = [20, 70, 300, 700, 1100]

def get_hard_mode_vector(text):
    # [핵심] 정답이 될만한 단어를 리스트에서 '강제 삭제' (데이터 수정 없이 코드에서 처리)
    # 이 단어들이 없으면 AI는 오직 '상황'만 보고 추리해야 하므로 난이도가 급상승함
    cheat_words = ['사기', '절도', '마약', '횡령', '폭행', '음주운전', '명예훼손', '교통사고', 
                   '공무집행방해', '강제추행', '사건', '혐의', '피고인', '판결', '징역', '무죄', 
                   '선고', '기소', '재판부', '상당', '피해', '발생']
    
    text = re.sub(r'[^\w\s]', '', text) # 특수문자 제거
    # 치트 단어가 아닌 것들만 남김
    words = [w for w in text.split() if w not in cheat_words and len(w) > 1]
    
    return set(words)

# 2. 데이터 로드 및 재분할 (1100개를 쓰기 위해 비율 조정)
if rank == 0:
    try:
        df = pd.read_csv('legal_data_total.csv')
        # 1100개까지 학습시키려면 학습용 데이터를 늘려야 함
        full_train_data = df.iloc[:1100].to_dict('records')   # 0~1100번 (학습용)
        test_data = df.iloc[1100:1200].to_dict('records')     # 1100~1200번 (테스트용 100개)
        challenge_data = df.iloc[1200:1600].to_dict('records')# 1200~1600번 (챌린지용 400개)
    except:
        print("❌ CSV 파일이 없습니다. generate_all_data.py를 먼저 실행하세요!")
        comm.Abort()
else:
    full_train_data = test_data = challenge_data = None

# 테스트 데이터 분산 (Scatter)
if rank == 0: test_chunks = np.array_split(test_data, size)
else: test_chunks = None
my_test_chunk = comm.scatter(test_chunks, root=0)

# 3. 5단계 반복 학습 시작
if rank == 0: print(f"🚀 AI 학습 시작: 5단계 난이도 상승 모드 (Cheat Words Removed)")

for i, data_count in enumerate(learning_phases):
    # Rank 0이 학습 데이터 양을 조절해서 잘라냄
    if rank == 0:
        current_train_data = full_train_data[:data_count]
    else:
        current_train_data = None
    
    # 모든 코어가 현재 단계의 학습 데이터를 공유
    current_train_data = comm.bcast(current_train_data, root=0)
    
    correct = 0
    for test_case in my_test_chunk:
        # 테스트 데이터도 똑같이 '어렵게(단어 삭제)' 만듦
        test_vec = get_hard_mode_vector(test_case['Facts'])
        best_cat, max_sim = "", -1
        
        # 학습 데이터와 비교
        for train_case in current_train_data:
            train_vec = get_hard_mode_vector(train_case['Facts'])
            
            # 유사도 계산
            if not (test_vec | train_vec): sim = 0
            else: sim = len(test_vec & train_vec) / len(test_vec | train_vec)
            
            if sim > max_sim: max_sim, best_cat = sim, train_case['Category']
        
        if best_cat == test_case['Category']: correct += 1
    
    # 결과 집계
    total_correct = comm.reduce(correct, op=MPI.SUM, root=0)
    
    if rank == 0:
        acc = total_correct / 100 # 테스트 데이터가 100개로 변경됨
        # 데이터가 적을 땐(20개) 점수가 낮고, 많을 땐(1100개) 점수가 높게 나옴
        print(f"🔄 Step {i+1} (Data: {data_count}ea) | Loss: {1-acc:.4f} | Accuracy: {acc*100:.2f}%")

# 4. 최종 챌린지 테스트
if rank == 0: 
    print("\n🏁 [최종 챌린지 테스트 (400개)]")
    chal_chunks = np.array_split(challenge_data, size)
else: chal_chunks = None

my_chal_chunk = comm.scatter(chal_chunks, root=0)
current_train_data = comm.bcast(full_train_data, root=0) # 1100개 전체 지식 사용

final_correct = 0
for test_case in my_chal_chunk:
    test_vec = get_hard_mode_vector(test_case['Facts'])
    best_cat, max_sim = "", -1
    for train_case in current_train_data:
        train_vec = get_hard_mode_vector(train_case['Facts'])
        if not (test_vec | train_vec): sim = 0
        else: sim = len(test_vec & train_vec) / len(test_vec | train_vec)
        if sim > max_sim: max_sim, best_cat = sim, train_case['Category']
    if best_cat == test_case['Category']: final_correct += 1

total_final = comm.reduce(final_correct, op=MPI.SUM, root=0)

if rank == 0:
    final_acc = total_final / 400
    print(f"🏆 Final Challenge Result | Loss: {1-final_acc:.4f} | Accuracy: {final_acc*100:.2f}%")
