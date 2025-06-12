import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# 데이터 인코딩 및 결측치 제거
def prepare_data(df_total):
    df_encoded = df_total.copy()

    # LightGBM 오류 방지를 위해 컬럼명에 있는 특수문자를 언더스코어로 변경
    df_encoded.columns = df_encoded.columns.str.replace(r"[^\w]", "_", regex=True)

    # 범주형 변수 인코딩
    categorical_cols = df_encoded.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        try:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        except:
            print(f"Encoding failed for: {col}")
    
    # 결측치 제거
    df_encoded = df_encoded.dropna()
    return df_encoded

# 모델 학습 및 성능 평가
def train_and_evaluate_models(df_encoded):
    print(f"전체 row 수: {len(df_encoded):,}")

    # 타겟 변수와 피처 분리
    X = df_encoded.drop(columns=['reordered', 'order_id', 'user_id', 'product_id'])
    y = df_encoded['reordered']

    # 학습/테스트 데이터 분리 (Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []

    # 공통 평가 함수 정의
    def evaluate_model(name, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 주요 평가지표 계산
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append([name, acc, roc, prec, rec, f1])
        return model

    # Random Forest
    evaluate_model('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42))

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )
    xgb_model = evaluate_model('XGBoost', xgb_model)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, random_state=42,
        min_data_in_leaf=10, min_split_gain=0.01
    )
    lgb_model = evaluate_model('LightGBM', lgb_model)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=3000, solver='lbfgs', random_state=42)
    evaluate_model('LogisticRegression', log_model)

    # 시각화 저장 디렉토리 생성
    os.makedirs("/opt/airflow/dags/plots", exist_ok=True)
    os.makedirs("/opt/airflow/dags/plots/modeling", exist_ok=True)

    # XGBoost 피처 중요도 시각화
    xgb_feat = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    xgb_feat.head(20).plot(kind='barh', color='teal')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("/opt/airflow/dags/plots/modeling/xgb_feature_importance.png")
    plt.close()

    # LightGBM 피처 중요도 시각화
    lgb_feat = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    lgb_feat = lgb_feat[lgb_feat.index != 'Unnamed__0']  # 불필요한 컬럼 제외
    plt.figure(figsize=(8, 5))
    sns.barplot(x=lgb_feat, y=lgb_feat.index, palette='magma')
    plt.title('LightGBM importance feature Top 5')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig("/opt/airflow/dags/plots/modeling/lgb_feature_importance.png")
    plt.close()

    # 모델별 성능 비교 결과 저장
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'ROC_AUC', 'Precision', 'Recall', 'F1'])
    results_df.to_csv('/opt/airflow/dags/plots/modeling/model_performance.csv', index=False)

    # 콘솔 출력
    print("\n모델 성능 비교 결과:")
    print(results_df)

# 전체 실행 함수 (Airflow 등에서 호출)
def run(df_total):
    df_encoded = prepare_data(df_total)
    train_and_evaluate_models(df_encoded)
