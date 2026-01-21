import pandas as pd
import random

random.seed(42) # 재현성을 위해 고정

locations = ['서울 강남구', '부산 해운대구', '대구 수성구', '인천 미추홀구', '광주 북구', '울산 남구', '대전 서구', '세종시']
evidences = ['CCTV 녹화 영상', '목격자 진술', '피고인의 자백', '휴대폰 포렌식 결과', '현장 DNA 채취 기록', '계좌 거래 내역', '통화 녹취록', 'IP 추적 결과']
mitigations = ['초범이며 깊이 반성하고 있는 점', '피해자와 원만히 합의한 점', '범행 가담 정도가 경미한 점', '범죄의 증거가 불충분하여 혐의를 입증하기 어려운 점']
aggravations = ['누범 기간 중 재범한 점', '피해 회복이 전혀 이루어지지 않은 점', '계획적이고 치밀한 범행 수법인 점', '수사 기관에서 허위 진술로 일관한 점']

scenarios = {
    '사기': ["고수익 투자를 미끼로 다수에게 자금을 편취함", "보이스피싱 조직의 현금 수거책으로 활동함", "인터넷 물품 거래 사기로 다수의 피해자를 양산함"],
    '절도': ["야간에 시정되지 않은 차량 문을 열고 금품을 절취함", "무인 매장의 키오스크를 파손하고 현금을 강취함", "백화점 의류 매장에서 고가의 의류를 가방에 숨겨 나옴"],
    '마약': ["SNS를 통해 필로폰을 매수하고 투약함", "특정 장소에 마약을 은닉하는 던지기 수법을 이용함", "클럽 등 유흥업소에서 향정신성 의약품을 교부함"],
    '횡령': ["업무상 보관하던 법인 자금을 개인 채무 변제에 사용함", "회사의 재고 물품을 임의로 반출하여 처분함", "동창회 총무로서 회비를 사적으로 유용함"],
    '폭행': ["음주 상태에서 행인과 시비가 붙어 안면부를 가격함", "운전 중 끼어들기 문제로 상대 운전자를 폭행함", "층간 소음 항의를 하러 온 이웃을 밀쳐 상해를 입힘"],
    '음주운전': ["혈중알코올농도 0.1% 상태로 10km 구간을 운전함", "음주 단속을 피해 도주하다가 가드레일을 충격함", "숙취가 해소되지 않은 상태로 운전대를 잡아 접촉사고를 냄"],
    '명예훼손': ["비방할 목적으로 허위 사실을 지역 커뮤니티에 게시함", "단톡방에서 특정인의 사생활에 대한 루머를 유포함", "경쟁 업체를 비방하는 악성 댓글을 조직적으로 작성함"],
    '교통사고': ["신호를 위반하여 횡단보도를 건너던 보행자를 충격함", "제한속도를 40km 초과하여 과속하다가 추돌사고를 냄", "어린이 보호구역 내에서 주의 의무를 위반하여 상해를 입힘"],
    '공무집행방해': ["출동한 경찰관의 멱살을 잡고 욕설을 함", "구청 민원실에서 집기를 집어던지며 난동을 부림", "음주 측정을 거부하고 경찰관을 차량으로 위협함"],
    '강제추행': ["혼잡한 출근길 지하철에서 피해자의 신체를 밀착함", "회식 자리에서 부하 직원의 의사에 반하여 신체를 만짐", "엘리베이터 내에서 기습적으로 피해자를 추행함"]
}

def generate_full_case(category, case_id):
    loc, scenario = random.choice(locations), random.choice(scenarios[category])
    evid = random.sample(evidences, 2)
    factor = random.choice(mitigations + aggravations)
    date, amt = f"2025년 {random.randint(1,12)}월 {random.randint(1,28)}일", f"{random.randint(100, 5000)}만 원"
    
    part1 = f"[{category} 사건 - {case_id}] 피고인은 {date}경 {loc} 일대에서 {scenario}의 혐의로 기소되었습니다. "
    part2 = f"상세 경위에 따르면 피고인은 {amt} 상당의 피해를 발생시켰으며, 범행 전후의 정황이 매우 불량합니다. "
    part3 = f"수사 기관은 {evid[0]} 및 {evid[1]}를 확보하여 유죄를 입증하였습니다. "
    part4 = f"본 법원은 피고인의 {factor} 등을 종합적으로 고려하여 판결을 내립니다. "
    outcome = "무죄" if "증거가 불충분" in factor else f"징역 {random.randint(1,3)}년"
    
    return part1 + part2 + part3 + part4 + f" [최종 판결: {outcome}]"

# 1,200개(Train/Test) + 400개(Challenge) = 총 1,600개 생성
categories = list(scenarios.keys())
all_cases = []
for i in range(1600):
    cat = categories[i % 10]
    all_cases.append({'Category': cat, 'Facts': generate_full_case(cat, f"CASE-{i}")})

pd.DataFrame(all_cases).to_csv('legal_data_total.csv', index=False, encoding='utf-8-sig')
print("✅ 1,600개 대용량 판례 데이터 생성 완료!")
