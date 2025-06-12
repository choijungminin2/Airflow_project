from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime
from scripts import preprocess, visualize, train_model, move_data  # ✅ 모듈 전체 import
import pandas as pd
import os

# DAG 기본 설정값
default_args = {
    'start_date': datetime(2025, 1, 1),
    'catchup': False  # 과거 스케줄 잡지 않도록 설정
}

# DAG 정의
with DAG(
    dag_id='instacart_etl_pipeline',                # DAG 이름
    schedule_interval='*/30 * * * *',               # 30분 간격 실행
    start_date=datetime(2025, 1, 1),                 # 시작 날짜
    catchup=False                                   # 과거 이력 catch-up 방지
) as dag:

    # 데이터 1000개 조건 검사 및 이동 함수
    def should_trigger():
        return move_data.check_and_move_1000_rows()

    # 조건 충족 시 다음 단계 실행 (ShortCircuitOperator)
    check_and_move = ShortCircuitOperator(
        task_id='check_and_move_1000_rows',
        python_callable=should_trigger
    )

    # 1단계: 데이터 전처리
    def preprocess_task():
        os.makedirs('/opt/airflow/dags/tmp', exist_ok=True)
        df_total, orders_train, order_products_train = preprocess.run()
        df_total.to_pickle('/opt/airflow/dags/tmp/df_total.pkl')  # 중간 결과 저장
        return '전처리 완료'

    # 2단계: 시각화
    def visualize_task():
        df_total = pd.read_pickle('/opt/airflow/dags/tmp/df_total.pkl')  # 저장된 전처리 결과 불러오기
        visualize.run_visualizations(df_total)
        return '시각화 완료'

    # 3단계: 모델 학습
    def train_model_task():
        df_total = pd.read_pickle('/opt/airflow/dags/tmp/df_total.pkl')  # 저장된 전처리 결과 불러오기
        train_model.run(df_total)
        return '모델링 완료'

    # 각 단계에 대해 PythonOperator 등록
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_task
    )

    task_visualize = PythonOperator(
        task_id='visualize_data',
        python_callable=visualize_task
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task
    )

    # Task 실행 순서 정의
    check_and_move >> task_preprocess >> task_visualize >> task_train
