import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 🏆 방금 확보한 '황금 데이터' 5개를 입력했습니다.
# =========================================================
real_accuracies = [42.0, 74.0, 95.0, 100.0, 99.0]
steps = ['Step 1\n(20ea)', 'Step 2\n(70ea)', 'Step 3\n(300ea)', 'Step 4\n(700ea)', 'Step 5\n(1100ea)']

# 로스율 자동 계산 (100 - 정확도)
real_losses = [(100 - acc) / 100 for acc in real_accuracies]

# 그래프 스타일 설정
plt.figure(figsize=(12, 6))
plt.style.use('default')

# -------------------------------------------------------
# 1. 로스율 (Loss Rate) - 빨간색 꺾은선 (우하향)
# -------------------------------------------------------
ax1 = plt.gca()
line1 = ax1.plot(steps, real_losses, color='#FF5252', marker='o', 
                 linestyle='-', linewidth=3, markersize=10, label='Loss Rate (Error)')

# 로스율 수치 표시
for i, v in enumerate(real_losses):
    ax1.text(i, v + 0.03, f"{v:.2f}", color='#FF5252', fontweight='bold', ha='center', fontsize=11)

ax1.set_ylabel('Loss Rate (0.0 ~ 1.0)', fontsize=12, fontweight='bold', color='#FF5252')
ax1.tick_params(axis='y', labelcolor='#FF5252')
ax1.set_ylim(0, 1.0)
ax1.set_xlabel('Training Data Scale (Knowledge Expansion)', fontsize=12, fontweight='bold')

# -------------------------------------------------------
# 2. 정확도 (Accuracy) - 파란색 막대 (우상향)
# -------------------------------------------------------
ax2 = ax1.twinx()
bar = ax2.bar(steps, real_accuracies, color='#448AFF', alpha=0.3, width=0.5, label='Accuracy (%)')

# 정확도 수치 표시
for i, v in enumerate(real_accuracies):
    ax2.text(i, v + 2, f"{v:.0f}%", color='#2962FF', fontweight='bold', ha='center', fontsize=11)

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#2962FF')
ax2.tick_params(axis='y', labelcolor='#2962FF')
ax2.set_ylim(0, 115)

# -------------------------------------------------------
# 3. 그래프 꾸미기 (제목 및 분석 주석)
# -------------------------------------------------------
plt.title('HPC AI Model Performance: Data Scaling Law', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# 핵심 분석 멘트 (그래프 위에 박스로 표시됨)
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax1.text(0, 0.45, "Insufficient Data\n(Underfitting)", fontsize=10, bbox=props, ha='center')
ax1.text(2, 0.30, "Rapid Learning\n(Scaling Law)", fontsize=10, bbox=props, ha='center')
ax1.text(4, 0.20, "Optimal Model\n(Generalized)", fontsize=10, bbox=props, ha='center')

# 범례 표시
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11)

# 저장
plt.tight_layout()
plt.savefig('final_result_graph.png', dpi=300)
print("✅ 최종 5단계 그래프(final_result_graph.png) 생성 완료! PPT에 넣으세요.")
