import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# [수정할 부분] 실험 결과를 콤마(,)로 구분해서 적으세요!
# ==========================================
# 프로세스 개수 (X축)
np_counts = [1, 3, 5, 7, 10, 12]

# 각 프로세스별 실행 시간 (초) -> 실험 후 여기에 채워넣으세요
# (지금은 예시 숫자입니다. 실제 측정값으로 바꾸세요!)
exec_times = [1.82, 0.65, 0.42, 0.31, 0.24, 0.19] 
# ==========================================

# 가속비(Speedup) 자동 계산 (T_1 / T_n)
t_serial = exec_times[0] # 첫 번째 값이 시리얼 시간이라고 가정
speedups = [t_serial / t for t in exec_times]

# 그래프 그리기 (2개를 위아래로 배치)
plt.figure(figsize=(10, 10))

# 1. 첫 번째 그래프: 실행 시간 (Execution Time)
plt.subplot(2, 1, 1) # 2행 1열 중 첫 번째
plt.plot(np_counts, exec_times, marker='o', linestyle='-', color='b', label='Execution Time')
plt.title(f'Execution Time vs Processes (NP)', fontsize=14, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(np_counts) # X축 눈금을 우리가 정한 NP 개수대로 찍기

# 값 표시하기
for x, y in zip(np_counts, exec_times):
    plt.text(x, y + 0.05, f"{y:.2f}s", ha='center', fontsize=10, fontweight='bold')

# 2. 두 번째 그래프: 가속비 (Speedup) - 이게 중요함!
plt.subplot(2, 1, 2) # 2행 1열 중 두 번째
plt.plot(np_counts, speedups, marker='s', linestyle='--', color='r', label='Speedup')

plt.plot(np_counts, np_counts, 'k:', alpha=0.3, label='Ideal Speedup') # 이상적인 선(점선)
plt.title('Speedup Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Number of Processes (NP)', fontsize=12)
plt.ylabel('Speedup (x times)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(np_counts)
plt.legend()

# 값 표시하기
for x, y in zip(np_counts, speedups):
    plt.text(x, y - 0.5, f"{y:.1f}x", ha='center', fontsize=10, fontweight='bold', color='red')

# 저장하기
plt.tight_layout()
plt.savefig('result_graph_curve.png', dpi=300)
print("완성! 'result_graph_curve.png' 파일이 생성되었습니다.")
