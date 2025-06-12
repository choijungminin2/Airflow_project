import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화 결과 저장 디렉토리 생성
os.makedirs("/opt/airflow/dags/plots", exist_ok=True)
os.makedirs("/opt/airflow/dags/plots/eda1", exist_ok=True)  # 히트맵 저장
os.makedirs("/opt/airflow/dags/plots/eda2", exist_ok=True)  # 재구매율 히스토그램 저장
os.makedirs("/opt/airflow/dags/plots/eda3", exist_ok=True)  # 유저-상품 주문비율 저장

# 요일 × 시간대별 주문량 히트맵
def plot_order_heatmap(df):
    # 요일과 시간대별 주문 수 집계
    df_heatmap = df.groupby(['order_dow', 'order_hour_of_day']).size().reset_index(name='order_count')
    # 피벗 테이블로 변환 (히트맵 그리기 위함)
    pivot_heatmap = df_heatmap.pivot(index='order_dow', columns='order_hour_of_day', values='order_count').fillna(0)

    # 시각화
    plt.figure(figsize=(16, 6))
    sns.heatmap(pivot_heatmap, cmap='YlOrRd', linewidths=0.3, linecolor='gray', annot=True, fmt='g')
    plt.title('Week × Hour Heatmap', fontsize=16)
    plt.xlabel('Hour')  # x축: 시간대
    plt.ylabel('Week')  # y축: 요일
    plt.xticks(rotation=0)
    plt.yticks(
        ticks=range(7),
        labels=['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'],
        rotation=0
    )
    plt.tight_layout()
    plt.savefig("/opt/airflow/dags/plots/eda1/order_heatmap.png")  # 이미지 저장
    plt.close()

# 고객별 평균 재구매율 히스토그램
def plot_user_reorder_rate(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['user_reorder_rate'], bins=30, kde=True, color='skyblue')  # KDE 곡선 포함
    plt.title('Avg. Reorder Rate per Customer')
    plt.xlabel('user_reorder_rate')  # x축: 평균 재구매율
    plt.ylabel('Number of Customers')  # y축: 고객 수
    plt.tight_layout()
    plt.savefig("/opt/airflow/dags/plots/eda2/user_reorder_rate.png")  # 이미지 저장
    plt.close()

# 유저-상품 주문 비율 히스토그램
def plot_user_product_order_ratio(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['user_product_order_ratio'], bins=30, kde=True, color='salmon')
    plt.title('Distribution of User-Product Order Ratio')
    plt.xlabel('Proportion ')  # x축: 해당 상품이 유저 전체 주문 중 차지한 비율
    plt.ylabel('Number of User-Product Pairs')  # y축: 유저-상품 조합 수
    plt.tight_layout()
    plt.savefig("/opt/airflow/dags/plots/eda3/user_product_order_ratio.png")  # 이미지 저장
    plt.close()

# 시각화 실행 함수 (Airflow 등에서 호출 가능)
def run_visualizations(df):
    plot_order_heatmap(df)
    plot_user_reorder_rate(df)
    plot_user_product_order_ratio(df)
