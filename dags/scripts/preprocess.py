import pandas as pd
import os

# 경로 설정
BASE_DIR = '/opt/airflow/dags'

# 1. CSV 불러오기
def load_csv():
    orders = pd.read_csv(f'{BASE_DIR}/data/orders.csv')
    aisles = pd.read_csv(f'{BASE_DIR}/data/aisles.csv')
    departments = pd.read_csv(f'{BASE_DIR}/data/departments.csv')
    order_products_prior = pd.read_csv(f'{BASE_DIR}/data/order_products__prior.csv')
    order_products_train = pd.read_csv(f'{BASE_DIR}/data/order_products__train.csv')
    products = pd.read_csv(f'{BASE_DIR}/data/products.csv')
    return orders, aisles, departments, order_products_prior, order_products_train, products

# 2. orders 전처리
def clean_orders(orders):
    orders['days_since_prior_order'].fillna(0, inplace=True)
    return orders

# 3. orders 분리
def split_orders(orders):
    orders_prior = orders[orders['eval_set'] == 'prior'].drop(columns=['eval_set'])
    orders_train = orders[orders['eval_set'] == 'train'].drop(columns=['eval_set'])
    return orders_prior, orders_train

# 4. order_products 전처리
def clean_order_products(df):
    if 'add_to_cart_order' in df.columns:
        return df.drop(columns=['add_to_cart_order'])
    return df

# 5. 메모리 문제로 total_csv는 따로 불러오기로함함
def total_csv():
    df_total = pd.read_csv(f'{BASE_DIR}/data/total.csv')
    return df_total

# 6. 피처 엔지니어링 함수
def generate_features(df_total):
    # 사용자별 평균 재구매율
    user_reorder_rate = df_total.groupby('user_id')['reordered'].mean().reset_index()
    user_reorder_rate.columns = ['user_id', 'user_reorder_rate']
    df_total = df_total.merge(user_reorder_rate, on='user_id', how='left')

    # 상품별 평균 재구매율
    product_reorder_rate = df_total.groupby('product_id')['reordered'].mean().reset_index()
    product_reorder_rate.columns = ['product_id', 'product_reorder_rate']
    df_total = df_total.merge(product_reorder_rate, on='product_id', how='left')

    # 사용자-상품 재구매 횟수
    user_product_reorder = df_total.groupby(['user_id', 'product_id'])['reordered'].sum().reset_index()
    user_product_reorder.columns = ['user_id', 'product_id', 'user_product_reorder_count']
    df_total = df_total.merge(user_product_reorder, on=['user_id', 'product_id'], how='left')

    # 유저-상품별 이전 주문 시점으로부터 경과일
    df_total = df_total.sort_values(['user_id', 'product_id', 'order_number'])
    df_total['days_since_product_last_order'] = df_total.groupby(['user_id', 'product_id'])['days_since_prior_order'].shift(1)
    df_total['days_since_product_last_order'] = df_total['days_since_product_last_order'].fillna(0)

    # 사용자별 총 주문 횟수
    user_order_count = df_total.groupby('user_id')['order_number'].max().reset_index()
    user_order_count.columns = ['user_id', 'user_total_orders']
    df_total = df_total.merge(user_order_count, on='user_id', how='left')

    # 사용자-상품 주문 비율
    user_product_orders = df_total.groupby(['user_id', 'product_id'])['reordered'].count().reset_index()
    user_total_orders = df_total.groupby('user_id')['order_number'].max().reset_index()
    user_product_orders.columns = ['user_id', 'product_id', 'user_product_orders']
    user_total_orders.columns = ['user_id', 'user_total_orders']
    df_ratio = user_product_orders.merge(user_total_orders, on='user_id')
    df_ratio['user_product_order_ratio'] = df_ratio['user_product_orders'] / df_ratio['user_total_orders']
    df_total = df_total.merge(df_ratio[['user_id', 'product_id', 'user_product_order_ratio']], on=['user_id', 'product_id'], how='left')

    return df_total

# 7. 전체 실행 함수 (메인)
def run():
    print("CSV 불러오기 시작")
    orders, aisles, departments, order_products_prior, order_products_train, products = load_csv()
    print("CSV 불러오기 완료")

    print("전처리 시작")
    orders = clean_orders(orders)
    order_products_prior = clean_order_products(order_products_prior)
    order_products_train = clean_order_products(order_products_train)
    print("전처리 완료")

    print("orders 분리")
    orders_prior, orders_train = split_orders(orders)

    print("병합 데이터 불러오기 시작")
    df_total = total_csv()
    print(f"병합 데이터 불러오기 완료, shape: {df_total.shape}")

    print("샘플링 시작")
    df_total = df_total.sample(n=5000, random_state=42)
    print(f"샘플링 완료, shape: {df_total.shape}")

    print("피처 엔지니어링 시작")
    df_total = generate_features(df_total)
    print(f"피처 엔지니어링 완료, 최종 shape: {df_total.shape}")

    return df_total, orders_train, order_products_train

