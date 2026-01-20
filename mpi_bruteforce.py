from mpi4py import MPI
import hashlib

def solve():
    # 1. MPI 초기화
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 2. 문제 설정 (Rank 0에서만 목표 설정)
    target_hash = None
    if rank == 0:
        # [시연용] 정답 설정 (6자리 숫자, 팀원들과 상의해서 바꾸세요!)
        secret_pin = "729431" 
        target_hash = hashlib.sha256(secret_pin.encode()).hexdigest()
        print(f"[Rank {rank}] 목표 해시 생성 완료: {target_hash[:10]}...", flush=True)

    # ==========================================
    # [배운 내용: Bcast] 1 -> N 통신
    # Rank 0의 target_hash를 모든 프로세스에 복사
    # ==========================================
    target_hash = comm.bcast(target_hash, root=0)

    # 3. 시간 측정 시작 (MPI 표준 시간 함수 Wtime 사용)
    # 모든 프로세스가 준비될 때까지 기다렸다가(Barrier) 시작하는 것이 더 정확함
    comm.Barrier() 
    start_time = MPI.Wtime()

    # ==========================================
    # [배운 내용: 블록 분배 (Block Distribution)]
    # 전체 100만 개를 12명이 공평하게 나누는 공식
    # ==========================================
    total_space = 1000000  # 000000 ~ 999999 탐색
    
    # 몫(count)과 나머지(remainder) 계산
    count = total_space // size
    remainder = total_space % size

    # 자신의 Rank에 맞는 시작(start)과 끝(end) 인덱스 계산
    if rank < remainder:
        start_idx = rank * (count + 1)
        end_idx = start_idx + count + 1
    else:
        start_idx = rank * count + remainder
        end_idx = start_idx + count

    # ==========================================
    # [배운 내용: 루프 (Loop)]
    # 할당받은 구역(Block)만 무차별 대입
    # ==========================================
    found_pw = None
    
    # 디버깅용: 내가 맡은 구역 출력 (너무 많이 출력되면 주석 처리)
    # print(f"[Rank {rank}] 탐색 시작: {start_idx} ~ {end_idx-1}", flush=True)

    for i in range(start_idx, end_idx):
        candidate = f"{i:06d}" # 숫자를 6자리 문자열로 (예: 1 -> "000001")
        
        # 해시 비교
        if hashlib.sha256(candidate.encode()).hexdigest() == target_hash:
            found_pw = candidate
            print(f"!!! [Rank {rank}] 암호 발견: {found_pw} !!!", flush=True)
            # 발견했다고 해서 바로 break 하지 않고 루프를 마저 돌거나 
            # 여기서는 간단히 break로 자기 할 일만 끝냄
            break 

    # ==========================================
    # [배운 내용: Gather] N -> 1 통신
    # 각자가 찾은 결과(없으면 None, 있으면 암호)를 Rank 0으로 수집
    # ==========================================
    all_results = comm.gather(found_pw, root=0)
    
    # 4. 시간 측정 종료 (모든 프로세스가 Gather에 도달해야 끝남)
    end_time = MPI.Wtime()

    # 5. 최종 결과 확인 (Rank 0만 수행)
    if rank == 0:
        final_answer = None
        for res in all_results:
            if res is not None:
                final_answer = res
                break
        
        duration = end_time - start_time
        
        print("\n" + "="*40, flush=True)
        if final_answer:
            print(f" 성공! 찾은 암호: {final_answer}", flush=True)
        else:
            print(f" 실패. 범위 내에 암호가 없습니다.", flush=True)
            
        print(f" 사용 프로세스 수: {size}개", flush=True)
        print(f" 총 소요 시간: {duration:.4f}초", flush=True)
        print("="*40 + "\n", flush=True)

if __name__ == "__main__":
    solve()
