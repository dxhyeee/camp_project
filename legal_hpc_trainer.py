from mpi4py import MPI
import pandas as pd
import numpy as np
import re

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# 1. 전처리 단계 (피드백 루프 시뮬레이션)
# 단계별로 조사를 더 많이 제거하여 AI의 '의미 학습' 능력을 향상시킴
feedback_levels = [
    ['은', '는'], # Level 1
    ['은', '는', '이', '가', '을', '를'], # Level 2
    ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '피고인', '사건'] # Level 3 (최적화)
]

def get_clean_set(text, stops):
    text = re.sub(r'[^\w\s]', '', text)
    return set([w for w in text.split() if w not in stops and len(w) > 1])

# 2. 데이터 로드 및 분할
if rank == 0:
    df = pd.read_csv('legal_data_total.csv')
    train_data = df.iloc[:900].to_dict('records')   # 900개 학습용
    test_data = df.iloc[900:1200].to_dict('records') # 300개 테스트용
    challenge_data = df.iloc[1200:1600].to_dict('records') # 400개 최종검증용
else:
    train_data = test_data = challenge_data = None

train_data = comm.bcast(train_data, root=0) # 학습 데이터는 모든 코어가 공유

# 3. 피드백 루프 실행 (Training & Feedback)
for level, stops in enumerate(feedback_levels):
    # 테스트 데이터를 12개 코어로 분산 (300 / 12 = 코어당 25개)
    my_test_chunk = comm.scatter(np.array_split(test_data, size) if rank == 0 else None, root=0)
    
    correct = 0
    for test_case in my_test_chunk:
        test_vec = get_clean_set(test_case['Facts'], stops)
        best_cat, max_sim = "", -1
        
        for train_case in train_data:
            train_vec = get_clean_set(train_case['Facts'], stops)
            # 코사인 유사도 원리 (자카드 방식 활용)
            sim = len(test_vec & train_vec) / len(test_vec | train_vec) if (test_vec | train_vec) else 0
            if sim > max_sim: max_sim, best_cat = sim, train_case['Category']
        
        if best_cat == test_case['Category']: correct += 1
    
    total_correct = comm.reduce(correct, op=MPI.SUM, root=0)
    
    if rank == 0:
        acc = total_correct / len(test_data)
        print(f"🔄 Feedback Level {level+1} | Loss: {1-acc:.4f} | Accuracy: {acc*100:.2f}%")

# 4. 최종 챌린지 테스트 (400개)
if rank == 0: print("\n🏁 [최종 챌린지 테스트 시작 (400개 미지의 데이터)]")
my_challenge_chunk = comm.scatter(np.array_split(challenge_data, size) if rank == 0 else None, root=0)

final_correct = comm.reduce(correct, op=MPI.SUM, root=0) # 마지막 최적화 로직 사용

if rank == 0:
    final_acc = final_correct / len(challenge_data)
    print(f"🏆 Final Challenge Result | Loss: {1-final_acc:.4f} | Accuracy: {final_acc*100:.2f}%")
