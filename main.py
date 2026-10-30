
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")

st.markdown("""
### 데이터 탐색하기

이 실습실에서는 뇌졸중 예측에 사용되는 데이터를 살펴봅니다.

사람들의 기본 정보, 건강 상태, 생활 습관 등을 담은 데이터를
확인하고 각 열의 의미를 직접 정리해 봅시다.

우리말 뜻과 데이터 출처는 직접 입력할 수 있으며,
입력한 내용은 자동으로 저장됩니다.
""")

# ==========================================
# 2. 데이터 불러오기
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "sihoo0815/data/refs/heads/main/stroke.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# ==========================================
# 3. 기록 파일 설정
# ==========================================

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "기록.json"

# 기록 폴더가 없으면 생성
RECORD_DIR.mkdir(parents=True, exist_ok=True)


def load_records():
    """저장된 JSON 기록 불러오기"""

    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

        except (json.JSONDecodeError, OSError):
            return {}

    return {}


def save_records(meanings, source):
    """우리말 뜻과 출처를 JSON 파일에 저장"""

    records = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=4
        )


# 기존 기록 불러오기
records = load_records()

saved_meanings = records.get("meanings", {})
saved_source = records.get("source", "")


# ==========================================
# 4. 데이터 기본 정보
# ==========================================

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


# ==========================================
# 5. 숫자 카드 4개
# ==========================================

st.divider()

st.subheader("📊 데이터 전체 살펴보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="👥 전체 사람 수",
    value=f"{total_people:,}명"
)

col2.metric(
    label="📋 전체 열 개수",
    value=f"{total_columns}개"
)

col3.metric(
    label="🧠 뇌졸중 환자 수",
    value=f"{stroke_count:,}명"
)

col4.metric(
    label="📈 뇌졸중 비율",
    value=f"{stroke_ratio:.2f}%"
)


# ==========================================
# 6. 열 이름 및 데이터 정보 표
# ==========================================

st.divider()

st.subheader("📝 데이터 열의 의미 알아보기")

st.markdown("""
각 열의 이름과 값의 종류, 빈 값 개수를 확인해 보세요.

**우리말 뜻** 칸은 교재를 참고하여 직접 작성할 수 있습니다.
작성한 내용은 자동으로 저장됩니다.
""")

# 열별 데이터 종류 구분
def get_value_type(column):

    if column == "id":
        return "식별자"

    elif pd.api.types.is_numeric_dtype(df[column]):
        return "숫자형"

    else:
        return "범주형"


column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        saved_meanings.get(col, "")
        for col in df.columns
    ],
    "값의 종류": [
        get_value_type(col)
        for col in df.columns
    ],
    "빈 값 개수": [
        int(df[col].isnull().sum())
        for col in df.columns
    ]
})

# 직접 편집 가능한 표
edited_info = st.data_editor(
    column_info,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="meaning_editor",
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            disabled=True
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻 (직접 입력)",
            help="교재를 참고하여 각 열의 뜻을 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            disabled=True
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            disabled=True
        )
    }
)

# ==========================================
# 7. 데이터 출처 입력
# ==========================================

st.divider()

st.subheader("📚 데이터 출처")

st.markdown("""
교재에 제시된 데이터 출처를 아래에 입력하세요.
입력한 내용은 자동으로 저장됩니다.
""")

source_text = st.text_area(
    "데이터 출처 입력",
    value=saved_source,
    height=120,
    placeholder="교재에 있는 데이터 출처를 입력하세요.",
    key="source_editor"
)


# ==========================================
# 8. 입력 내용 자동 저장
# ==========================================

current_meanings = {}

for _, row in edited_info.iterrows():
    current_meanings[row["열 이름"]] = row["우리말 뜻"]

try:
    save_records(
        current_meanings,
        source_text
    )

except OSError as e:
    st.error(f"기록 저장 중 오류가 발생했습니다: {e}")


# ==========================================
# 9. 데이터 처음 5행 표시
# ==========================================

st.divider()

st.subheader("🔍 데이터 원본 확인하기")

st.markdown("""
실제 데이터를 처음 다섯 줄만 확인해 봅시다.

아래 표는 원본 데이터를 수정하지 않고 그대로 보여 줍니다.
""")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 10. 마무리
# ==========================================

st.divider()

st.caption("🧠 뇌졸중 예측 실습실 | 데이터 탐색 학습 공간")

