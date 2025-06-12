# Instacart ETL 자동화 프로젝트

Instacart 주문 데이터를 활용하여 데이터 수집부터 전처리, 시각화, 머신러닝 예측까지 자동화된 ETL 파이프라인을 구축한 프로젝트입니다.  
Apache Airflow를 기반으로 하여 이커머스 데이터 처리의 전 과정을 자동화하였으며, 주기적인 데이터 적재 및 분석이 가능하도록 설계되었습니다.

---


## 파이프라인 구성 요약

1. **데이터 자동화 설정 (`move_data.py`)**  
   - `orders.csv`에서 1000개씩 샘플링하여 `total_new.csv`로 순차 저장  
   - Airflow DAG이 새로운 데이터 감지 시 파이프라인 실행

2. **전처리 (`preprocess.py`)**  
   - CSV 불러오기 및 결합 (`orders`, `products`, `aisles`, `departments`)  
   - 사용자/상품 기반 파생 변수 생성 (재구매율, 비율 등)

3. **시각화 (`visualize.py`)**  
   - 요일/시간대별 주문 히트맵  
   - 고객별 평균 재구매율, 유저-상품 주문 비율 분포 등 시각화

4. **모델 학습 (`train_model.py`)**  
   - 인코딩 및 결측치 처리  
   - 모델 비교: Random Forest, XGBoost, LightGBM, Logistic Regression  
   - Accuracy, ROC AUC, Precision, Recall, F1 등 지표 출력 및 저장  
   - 피처 중요도 시각화 (XGBoost, LightGBM 기준)

5. **Airflow DAG (`etl_workflow.py`)**  
   - `move_data` 조건 충족 시 파이프라인 실행  
   - `preprocess` → `visualize` → `train_model` 순차 실행 자동화

---

## 기대 효과

- CSV 데이터를 자동으로 분할하고 처리할 수 있는 파이프라인 구현  
- 사용자 재구매 패턴 기반 예측 모델링으로 추천 시스템의 기반 마련  
- 다양한 모델 간 성능 비교로 최적 모델 도출 가능  
- Airflow 기반 워크플로우 자동화로 운영 효율성 및 확장성 향상
---

## 사용 기술 스택
- **데이터 처리**: Python, Pandas  
- **시각화**: Matplotlib, Seaborn  
- **모델링**: Scikit-learn, XGBoost, LightGBM  
- **워크플로우 자동화**: Apache Airflow  
