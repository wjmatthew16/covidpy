# covidpy
이 프로젝트는 SEIR 모델을 기반으로 한 코로나19 감염병 시뮬레이션 프로그램이다. 코로나19로 지난 몇년간 많은 사람들이 힘든 시기를 보냈고, 감염에 대한 불안감을 느꼈다. 이러한 상황에서 감염에 대한 모델링을 통해 집단의 감염 형태를 모델링할 수 있다면 큰 도움이 되리라 생각한다. 이 프로그램 단순한 전염 확률이 아닌 현실에서 실제로 영향을 미치는 변수들, 즉 사람의 연령, 마스크 종류, 거리, 접촉 시간 등을 정량화하여 감염 확산 양상을 더 실제적으로 모사하는 데에 있다. 단순히 감염자 수를 예측하는 것이 아니라, 이들이 어떤 조건에서 감염되고 어떤 조건에서 전파가 억제되는지를 확인하는 것이 목표였다.

입력은 input.txt 파일을 통해 이루어진다. 첫 번째 줄에는 사람 수 n이 주어지며, 두 번째 줄에는 초기 감염자 수 k, 세 번째 줄에는 초기 감염자들의 번호가 공백으로 구분되어 입력된다. 네 번째 줄에는 각 사람의 실제 나이가 순서대로 주어지는데, 이는 내부적으로 자동 분류되어 감염 민감도 계수로 변환된다. 이후 시뮬레이션을 며칠간 수행할지 나타내는 정수 days가 주어지고, 그다음 줄에는 총 접촉 횟수 m이 주어진다. 나머지 줄은 각 접촉을 나타내며, 각각의 줄은 "날짜 사람1 사람2 마스크1 마스크2 거리 시간" 형태로 이루어진다. 예를 들어, "2 1 3 x k 1.5 10.0"은 2일 차에 1번 사람과 3번 사람이 각각 마스크 미착용(x)과 KF94(k)를 착용한 상태에서 1.5m 거리로 10분간 접촉했다는 의미이다.

마스크는 다음과 같은 문자를 통해 구분되며, 문자 조합에 따라 감염률이 달라지도록 설정되어 있다:

x: 미착용 (none)

c: 천 마스크 (cloth)

s: 수술용 마스크 (surgical)

k: KF94 또는 N95급 마스크

이 조합에 따라 기본 전염 확률은 테이블로 정의되어 있다. 예를 들어, 두 사람이 모두 마스크를 쓰지 않은 경우(x, x) 기본 감염률은 0.9로 설정되며, KF94 착용자끼리 접촉한 경우(k, k)는 0.01로 낮다. 감염률은 아래와 같이 설정되어 있다:

마스크 조합	감염률

x - x	0.90

x - c	0.70

x - s	0.50

x - k	0.20

c - x	0.40

s - x	0.20

k - x	0.10

c - c	0.50

s - k	0.03

k - k	0.01


이처럼 마스크의 착용 여부와 종류에 따라 전파 차단 효과를 반영하였으며, 데이터는 실제 실험 및 정책 보고서에 기반하여 대략적인 현실 감각을 반영하였다. 이중 x는 전혀 보호 효과가 없는 상태이며, k는 가장 강한 보호력을 가진 고성능 마스크로 간주된다. 마스크 조합이 테이블에 존재하지 않는 경우에는 대칭 항목을 참조하거나 기본값 0.5를 사용한다.

연령에 따라 감염 민감도 또한 다르게 적용하였다. 각 개인의 실제 나이를 입력으로 받아 다음과 같이 6단계의 민감도 그룹으로 자동 분류된다:

나이대	그룹 번호	감염 민감도 계수

20대 이하	1	0.7

30대	2	0.8

40대	3	1.0

50대	4	1.2

60대	5	1.5

70대 이상	6	1.8


이 계수는 동일한 상황에서도 고령자의 감염 확률이 더 높아지는 효과를 반영한다. 따라서 동일한 시간과 거리에서의 접촉이라 하더라도, 고령자는 더 높은 확률로 감염될 수 있다. 이는 연령에 따른 면역력 차이, 기저질환 유무 등을 간접적으로 고려하기 위한 설계이다.

실제로 감염 확률은 다음의 수식을 통해 계산된다:

P = mask_prob × exp(-α × d) × (1 - exp(-β × t)) × age_multiplier

여기서 mask_prob은 위에서 설명한 마스크 조합별 기본 전염 확률이고, d는 두 사람 간 거리(m), t는 접촉 시간(분)이다. α와 β는 감염률 민감도 조절 계수로 각각 1.0, 0.2로 설정하였다. 거리가 멀어질수록 전염 확률이 급격히 낮아지고, 시간은 일정 이상부터 감염 확률이 급격히 상승하게 된다. age_multiplier는 감염 대상자의 연령 민감도 계수이며, 위 표에 따라 결정된다.

본 시뮬레이션에서 SEIR 모델의 상태 전이는 감염 후 일정한 시간 경과에 따라 단계적으로 이루어진다. 감염되지 않은 상태인 S(Susceptible)는 감염자와 접촉했을 때 일정 확률로 노출 상태인 E(Exposed)로 전이된다. 이때 S에서 E로의 전이는 시간 지연 없이 즉시 이루어지며, 감염 확률은 마스크 종류, 거리, 접촉 시간, 감염 대상자의 연령 등을 고려하여 결정된다.

E 상태로 전이된 사람은 실제로 전염력을 가지지는 않지만, 일정 시간이 지나면 감염 상태인 I(Infectious)로 변하게 된다. 코드에서는 이 노출 기간(exposure period)을 2일로 설정하였다. 즉, 감염자로부터 노출된 지 2일이 지나면 해당 인원은 실제로 다른 사람에게 감염을 전파할 수 있는 전염 상태로 바뀐다.

I 상태로 전이된 사람은 이후 5일간 전염력을 가지며, 그 이후에는 자연스럽게 회복 상태인 R(Recovered)로 전이된다. 이 감염 지속 기간(infectious period)은 코드 상에서 5일로 설정되어 있다. 따라서 하나의 감염 전파가 일어나고 나서 최종적으로 R 상태에 이르기까지의 최소 총 경과 시간은 2일의 노출 기간과 5일의 감염 기간을 합쳐 총 7일이다.

출력은 두 가지 파일로 이루어진다. 첫 번째는 output.txt이며, 시뮬레이션을 1000회 반복 수행한 뒤 날짜별로 S, E, I, R 상태에 해당하는 인원 수의 평균값을 CSV 형식으로 저장한다. 두 번째는 output_graph.png로, S, E, I, R의 평균 경로를 선 그래프로 나타내고, 각 선에는 시뮬레이션 표준편차에 해당하는 음영 영역이 함께 표시된다. 이를 통해 감염 추세뿐만 아니라 통계적 신뢰구간까지 한눈에 파악할 수 있다.

전체 코드의 흐름은 다음과 같다. 먼저 입력 파일을 열어 각 정보를 파싱한 뒤, 나이 정보를 기반으로 각 사람을 연령대 그룹으로 분류한다. 각 연령 그룹은 감염에 대한 민감도 계수를 다르게 부여받으며, 이 값은 감염 확률 계산의 가중치로 사용된다. 이후 접촉 정보를 날짜별로 정리하고, 그 데이터를 기반으로 1000회의 독립 시뮬레이션을 수행한다. 시뮬레이션은 상태 배열을 통해 각 사람의 상태를 'S', 'E', 'I', 'R' 중 하나로 관리하며, 노출 기간이 지나면 'E'에서 'I'로, 감염 기간이 지나면 'I'에서 'R'로 자동 전이된다. 각 날마다 모든 접촉에 대해 감염자가 비감염자에게 전파할 가능성을 계산하며, 이때 감염 확률은 마스크 조합, 거리, 시간, 연령 민감도를 반영한 수식에 따라 결정된다. 전파 확률은 지수함수를 사용하여 거리와 시간에 따라 감소 또는 증가하며, 마스크 조합별로 설정된 감염 기본 확률값에 곱해져 실제 감염 가능성이 산출된다. 이 확률이 난수보다 크면 감염이 발생한다는 방식으로 상태 전이가 이루어진다.

시뮬레이션이 종료되면 결과값은 NumPy 배열로 저장되며, Pandas를 통해 일자별 평균값과 표준편차를 계산하여 텍스트와 이미지로 각각 출력된다. 그래프와 텍스트는 자동으로 저장된다. 전체 코드 구조는 반복 실험을 효율적으로 수행할 수 있도록 벡터 연산 기반으로 설계되었으며, 변수들을 재설정하여 추후에 감염확률, 완치까지의 기간의 변경 또는 새로운 전염병에 대해서도 적용할 수 있도록 설계되었다. 

함께 첨부한 input.txt 파일은 100명의 사람, 35명의 임의의 감염자를 바탕으로 30일간의 접촉 기록을 담고 있는 예제 실행 파일이다. 또한 output.txt, output_graph.png는 input.txt를 바탕으로 실행한 예제 출력파일이다. 




