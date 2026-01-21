from mpi4py import MPI
import pandas as pd
import numpy as np
import re

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# 전략 변경: 전처리 수준은 고정하되, '공부하는 데이터 양'을 늘림
# 학습 단계: [조금 공부, 적당히 공부, 많이 공부]
learning_phases = [50, 200, 900] 

def get_clean_set(text):
    # 기본적인 전처리 적용
    stops = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '피고인', '사건', '판결']
    text = re.sub(r'[^\w\s]', '', text)
    return set([w for w in text.split() if w not in stops and len(w) > 1])

# 1. 데이터 로드
if rank == 0:
    try:
        df = pd.read_csv('legal_data_total.csv')
        full_train_data = df.iloc[:900].to_dict('records')   
        test_data = df.iloc[900:1200].to_dict('records') 
        challenge_data = df.iloc[1200:1600].to_dict('records')
    except:
        print("CSV 파일이 없습니다. generate_all_data.py를 먼저 실행하세요.")
        comm.Abort()
else:
    full_train_data = test_data = challenge_data = None

# 테스트 데이터는 미리 분산
if rank == 0:
    test_chunks = np.array_split(test_data, size)
else:
    test_chunks = None
    
my_test_chunk = comm.scatter(test_chunks, root=0)

# 2. 단계별 학습 (데이터 양 증가)
for i, data_count in enumerate(learning_phases):
    # Rank 0이 학습 데이터 양을 조절해서 뿌림
    if rank == 0:
        current_train_data = full_train_data[:data_count]
    else:
        current_train_data = None
    
    # 모든 코어가 현재 단계의 학습 데이터를 공유받음
    current_train_data = comm.bcast(current_train_data, root=0)
    
    correct = 0
    for test_case in my_test_chunk:
        test_vec = get_clean_set(test_case['Facts'])
        best_cat, max_sim = "", -1
        
        # 학습 데이터와 비교
        for train_case in current_train_data:
            train_vec = get_clean_set(train_case['Facts'])
            # 유사도 계산
            if not (test_vec | train_vec): sim = 0
            else: sim = len(test_vec & train_vec) / len(test_vec | train_vec)
            
            if sim > max_sim: max_sim, best_cat = sim, train_case['Category']
        
        if best_cat == test_case['Category']: correct += 1
    
    # 결과 집계
    total_correct = comm.reduce(correct, op=MPI.SUM, root=0)
    
    if rank == 0:
        acc = total_correct / 300 # 테스트 데이터 300개 기준
        print(f"🔄 Learning Phase {i+1} (Data: {data_count}ea) | Loss: {1-acc:.4f} | Accuracy: {acc*100:.2f}%")

# 3. 최종 챌린지 테스트
if rank == 0: 
    print("\n🏁 [최종 챌린지 테스트 (400개)]")
    chal_chunks = np.array_split(challenge_data, size)
else:
    chal_chunks = None

my_chal_chunk = comm.scatter(chal_chunks, root=0)
current_train_data = comm.bcast(full_train_data, root=0) # 전체 데이터로 검증

final_correct = 0
for test_case in my_chal_chunk:
    test_vec = get_clean_set(test_case['Facts'])
    best_cat, max_sim = "", -1
    for train_case in current_train_data:
        train_vec = get_clean_set(train_case['Facts'])
        if not (test_vec | train_vec): sim = 0
        else: sim = len(test_vec & train_vec) / len(test_vec | train_vec)
        if sim > max_sim: max_sim, best_cat = sim, train_case['Category']
    if best_cat == test_case['Category']: final_correct += 1

total_final = comm.reduce(final_correct, op=MPI.SUM, root=0)

if rank == 0:
    final_acc = total_final / 400
    print(f"🏆 Final Challenge Result | Loss: {1-final_acc:.4f} | Accuracy: {final_acc*100:.2f}%")
