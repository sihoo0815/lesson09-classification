
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 이해하고, 분석하고, 뇌졸중 예측을 실습하는 공간입니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

SAVE_DIR = Path("내_기록")
SAVE_FILE = SAVE_DIR / "stroke_record.json"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 저장된 기록 불러오기
# --------------------------------------------------

def load_record():
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


record = load_record()

saved_meanings = record.get("meanings", {})
saved_source = record.get("source", "")


# --------------------------------------------------
# 4. 기록 저장하기
# --------------------------------------------------

def save_record(meanings, source):
    data = {
        "meanings": meanings,
        "source": source
    }

    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


# --------------------------------------------------
# 5. 데이터 기본 정보
# --------------------------------------------------

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


# --------------------------------------------------
# 6. 주요 통계 카드
# --------------------------------------------------

st.header("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 전체 사람 수", f"{total_people:,}명")
col2.metric("📋 열 개수", f"{total_columns}개")
col3.metric("🧠 뇌졸중 발생 인원", f"{stroke_count:,}명")
col4.metric("📈 뇌졸중 발생 비율", f"{stroke_ratio:.2f}%")


st.divider()


# --------------------------------------------------
# 7. 데이터 소개
# --------------------------------------------------

st.header("📚 데이터 소개")

st.markdown("""
이 데이터는 사람들의 건강 상태와 생활 습관 등의 정보를 담고 있으며,
뇌졸중 발생 여부를 분석하고 예측하는 머신러닝 실습에 활용할 수 있습니다.

각 행은 한 사람의 정보를 나타내며, 각 열은 해당 사람의 특성을 나타냅니다.

- 입력 데이터: 건강 상태 및 생활 습관 관련 정보
- 예측 대상: `stroke`
- 예측 대상의 값: 0 또는 1
- 결측값: BMI 등의 일부 항목에 빈 값이 존재합니다.
""")


# --------------------------------------------------
# 8. 열 정보 정리
# --------------------------------------------------

st.subheader("📝 데이터 열 정보 정리")

st.info(
    "우리말 뜻 칸을 직접 입력해 보세요. "
    "입력한 내용은 자동으로 저장됩니다."
)

column_info = []

for column in df.columns:
    unique_count = df[column].nunique(dropna=True)
    missing_count = int(df[column].isna().sum())

    if pd.api.types.is_numeric_dtype(df[column]):
        value_type = f"숫자형 (고유값 {unique_count}개)"
    else:
        value_type = f"범주형 (고유값 {unique_count}개)"

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_type,
        "빈 값 개수": missing_count
    })


info_df = pd.DataFrame(column_info)

edited_info = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="column_editor",
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻 (직접 입력)",
            width="large"
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    }
)

current_meanings = {
    row["열 이름"]: row["우리말 뜻"]
    for _, row in edited_info.iterrows()
}

current_source = st.session_state.get(
    "source_input",
    saved_source
)

try:
    save_record(current_meanings, current_source)
except OSError as e:
    st.error(f"기록 저장에 실패했습니다: {e}")


st.divider()


# --------------------------------------------------
# 9. 데이터 미리보기
# --------------------------------------------------

st.header("🔍 데이터 미리보기")

st.markdown("실제 데이터의 첫 다섯 행입니다.")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# --------------------------------------------------
# 10. 데이터 출처
# --------------------------------------------------

st.header("📖 데이터 출처")

st.markdown("""
교재에 제시된 데이터 출처를 아래에 직접 적어 보세요.
작성한 내용은 JSON 파일에 자동으로 저장됩니다.
""")

source_text = st.text_area(
    "데이터 출처 입력",
    value=saved_source,
    placeholder="교재를 참고하여 데이터 출처를 입력하세요.",
    height=120,
    key="source_input"
)

try:
    save_record(current_meanings, source_text)
except OSError as e:
    st.error(f"기록 저장에 실패했습니다: {e}")


st.caption(f"데이터 원본: {DATA_URL}")

st.success(
    "우리말 뜻과 데이터 출처는 "
    "`내_기록/stroke_record.json` 파일에 저장됩니다."
)

st.divider()

st.caption("🧠 뇌졸중 예측 실습실 | 데이터 이해부터 머신러닝까지")
