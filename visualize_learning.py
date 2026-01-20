import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# ⚠️ [중요] MPI 실행 결과(터미널 출력값)를 여기에 입력하세요!
# 예: 터미널에 "Accuracy: 35.5%"라고 떴으면 35.5를 입력
# ==============================================================================
real_accuracies = [35.5, 78.2, 99.1]  # <--- 이 숫자를 본인의 결과로 수정하세요! (추가도 가능)

# 로스율 자동 계산 (100% - 정확도)
real_losses = [(100 - acc) / 100 for acc in real_accuracies]
steps = ['Step 1\n(Basic)', 'Step 2\n(Feedback)', 'Step 3\n(Final)']

# 그래프 스타일 설정
plt.figure(figsize=(12, 6))
plt.style.use('default') # 기본 스타일 사용 (깔끔함)

# -------------------------------------------------------
# 1. 로스율 (Loss Rate) - 빨간색 꺾은선
# -------------------------------------------------------
ax1 = plt.gca()
line1 = ax1.plot(steps, real_losses, color='#FF5252', marker='o', 
                 linestyle='-', linewidth=3, markersize=10, label='Loss Rate (Error)')

# 값 표시 (Annotation)
for i, v in enumerate(real_losses):
    ax1.text(i, v + 0.05, f"{v:.2f}", color='#FF5252', fontweight='bold', ha='center')

ax1.set_ylabel('Loss Rate (0.0 ~ 1.0)', fontsize=12, fontweight='bold', color='#FF5252')
ax1.tick_params(axis='y', labelcolor='#FF5252')
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('Feedback Loop Iterations', fontsize=12, fontweight='bold')

# -------------------------------------------------------
# 2. 정확도 (Accuracy) - 파란색 막대
# -------------------------------------------------------
ax2 = ax1.twinx() # Y축 공유
bar = ax2.bar(steps, real_accuracies, color='#448AFF', alpha=0.3, width=0.4, label='Accuracy (%)')

# 값 표시
for i, v in enumerate(real_accuracies):
    ax2.text(i, v + 2, f"{v:.1f}%", color='#2962FF', fontweight='bold', ha='center')

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#2962FF')
ax2.tick_params(axis='y', labelcolor='#2962FF')
ax2.set_ylim(0, 110)

# -------------------------------------------------------
# 3. 설명 주석 (이게 핵심!) - 왜 좋아졌는지 설명
# -------------------------------------------------------
plt.title('AI Model Training Process: Loss Convergence Analysis', fontsize=16, pad=20)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# 단계별 핵심 내용 화살표로 설명
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax1.text(0, 0.2, "High Noise\n(No Preprocessing)", fontsize=9, bbox=props, ha='center')
ax1.text(1, 0.2, "Feature Extraction\n(Removing Particles)", fontsize=9, bbox=props, ha='center')
ax1.text(2, 0.2, "Optimization\n(Loss ≈ 0)", fontsize=9, bbox=props, ha='center')

# 범례 표시
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# 저장
plt.tight_layout()
plt.savefig('final_result_graph.png', dpi=300)
print(f"✅ 결과 값을 반영한 최종 그래프(final_result_graph.png) 생성 완료!")
print(f"   - 입력된 정확도: {real_accuracies}")
print(f"   - 계산된 로스율: {real_losses}")
