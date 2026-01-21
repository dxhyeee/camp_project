from mpi4py import MPI
import hashlib
import random  # 랜덤 모듈 추가

def solve():
    # 1. MPI 초기화
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 2. 문제 설정 (Rank 0에서만 목표 설정)
    target_hash = None
    
    if rank == 0:
        # [수정됨] 0부터 99,999,999 사이의 숫자를 랜덤으로 뽑음
        random_num = random.randint(0, 99999999)
        
        # 8자리 문자열로 변환 (예: 123 -> "00000123")
        secret_pin = f"{random_num:08d}"
        
        # 해시 생성 (이것만 다른 친구들에게 알려줌)
        target_hash = hashlib.sha256(secret_pin.encode()).hexdigest()
        
        print(f"\n[Rank {rank}] 🎲 랜덤 암호 생성 완료! (정답은 비밀 쉿!)", flush=True)
        print(f"[Rank {rank}] 목표 해시값: {target_hash[:10]}...", flush=True)
        
        # (테스트용) 정답을 미리 보고 싶으면 아래 주석을 푸세요
        # print(f"[Debug] 실제 정답: {secret_pin}", flush=True)

    # 3. 목표 해시값 전파 (Bcast) -> "자, 이 해시값을 가진 숫자를 찾아봐!"
    target_hash = comm.bcast(target_hash, root=0)

    # 4. 준비 및 시간 측정 시작
    comm.Barrier() 
    start_time = MPI.Wtime()

    # ==========================================
    # 탐색 범위 1억 개 (00000000 ~ 99999999)
    # ==========================================
    total_space = 100000000 
    
    # 1/N 로 일감 분배
    count = total_space // size
    remainder = total_space % size

    if rank < remainder:
        start_idx = rank * (count + 1)
        end_idx = start_idx + count + 1
    else:
        start_idx = rank * count + remainder
        end_idx = start_idx + count

    # 자기 구역 탐색
    found_pw = None
    
    for i in range(start_idx, end_idx):
        candidate = f"{i:08d}" 
        
        # 해시 비교
        if hashlib.sha256(candidate.encode()).hexdigest() == target_hash:
            found_pw = candidate
            print(f"!!! [Rank {rank}] 🔓 암호 발견: {found_pw} !!!", flush=True)
            break 

    # 5. 결과 취합 (Gather)
    all_results = comm.gather(found_pw, root=0)
    
    end_time = MPI.Wtime()

    # 6. 최종 결과 출력
    if rank == 0:
        final_answer = None
        for res in all_results:
            if res is not None:
                final_answer = res
                break
        
        duration = end_time - start_time
        
        print("\n" + "="*50, flush=True)
        if final_answer:
            print(f" ✅ 성공! 컴퓨터가 숨긴 암호: {final_answer}", flush=True)
        else:
            print(f" ❌ 실패. (혹시 범위 설정이 잘못되었나요?)", flush=True)
            
        print(f" 💻 참여 프로세스 수: {size}개", flush=True)
        print(f" ⏱️ 총 소요 시간: {duration:.4f}초", flush=True)
        print("="*50 + "\n", flush=True)

if __name__ == "__main__":
    solve()
