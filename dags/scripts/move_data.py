import os
import pandas as pd  # pandas도 사용하고 있으므로 꼭 필요

# 필요한 경로
SOURCE_PATH = "/opt/airflow/dags/data/total.csv"
DEST_PATH = "/opt/airflow/dags/data/total_new.csv"
TEMP_SAMPLE_PATH = "/opt/airflow/dags/data/temp_1000.csv"
OFFSET_TRACKER = "/opt/airflow/dags/data/offset.txt"

def check_and_move_1000_rows():
    try:
        if not os.path.exists(DEST_PATH):
            df_sample = pd.read_csv(SOURCE_PATH, nrows=1000)
            df_sample.to_csv(DEST_PATH, index=False)
            df_sample.to_csv(TEMP_SAMPLE_PATH, index=False)

            with open(OFFSET_TRACKER, 'w') as f:
                f.write("1000")

            print("처음 1000개 저장 완료")
            return True

        offset = 0
        if os.path.exists(OFFSET_TRACKER):
            with open(OFFSET_TRACKER, 'r') as f:
                offset = int(f.read().strip())

        df_sample = pd.read_csv(SOURCE_PATH, skiprows=range(1, offset + 1), nrows=1000)

        if df_sample.empty:
            print("새로운 데이터 없음")
            return False

        df_existing = pd.read_csv(DEST_PATH)
        df_sample.to_csv(TEMP_SAMPLE_PATH, index=False)
        df_combined = pd.concat([df_existing, df_sample], ignore_index=True)
        df_combined.to_csv(DEST_PATH, index=False)

        with open(OFFSET_TRACKER, 'w') as f:
            f.write(str(offset + len(df_sample)))

        print(f"{len(df_sample)}개 이동 완료 (누적: {offset + len(df_sample)})")
        return True

    except Exception as e:
        print(f"오류 발생: {e}")
        return False
