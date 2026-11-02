
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 데이터 탐색")
st.caption("그래프와 표를 이용하여 뇌졸중 데이터를 탐색합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


for column in [
    "age", "avg_glucose_level", "bmi",
    "stroke", "hypertension", "heart_disease"
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ==================================================
# 1. 나이와 평균 혈당 분포
# ==================================================

st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
        color_discrete_sequence=["#4C78A8"]
    )
    fig.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#F58518"]
    )
    fig.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )
    st.plotly_chart(fig, use_container_width=True)


st.divider()


# ==================================================
# 2. 뇌졸중 여부에 따른 상자그림
# ==================================================

st.header("2. 뇌졸중 여부에 따른 비교")

compare_df = df[df["stroke"].isin([0, 1])].copy()

compare_df["뇌졸중 여부"] = compare_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        compare_df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 나이",
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        },
        labels={"age": "나이", "뇌졸중 여부": "구분"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        compare_df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 평균 혈당",
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        },
        labels={
            "avg_glucose_level": "평균 혈당",
            "뇌졸중 여부": "구분"
        }
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


mean_table = (
    compare_df
    .groupby("뇌졸중 여부")[
        ["age", "avg_glucose_level"]
    ]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
    .rename(columns={
        "뇌졸중 여부": "구분",
        "age": "평균 나이",
        "avg_glucose_level": "평균 혈당"
    })
    .round(2)
)

st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)


st.divider()


# ==================================================
# 3. 고혈압과 심장병에 따른 비율
# ==================================================

st.header("3. 질병 여부에 따른 뇌졸중 비율")


def calculate_rate(data, column, name):
    valid = data[
        data[column].isin([0, 1])
        & data["stroke"].isin([0, 1])
    ].copy()

    result = (
        valid.groupby(column)
        .agg(
            전체_인원=("stroke", "size"),
            뇌졸중_인원=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중_비율"] = (
        result["뇌졸중_인원"]
        / result["전체_인원"] * 100
    )

    result["구분"] = result[column].map({
        0: f"{name} 없음",
        1: f"{name} 있음"
    })

    return result


def draw_rate_chart(result, title):
    fig = px.bar(
        result,
        x="구분",
        y="뇌졸중_비율",
        text=result["뇌졸중_비율"].map(
            lambda x: f"{x:.2f}%"
        ),
        title=title,
        color="구분",
        color_discrete_sequence=["#4C78A8", "#E45756"],
        labels={
            "구분": "질병 여부",
            "뇌졸중_비율": "뇌졸중 비율 (%)"
        }
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_title="뇌졸중 비율 (%)"
    )
    return fig


hyper_result = calculate_rate(df, "hypertension", "고혈압")
heart_result = calculate_rate(df, "heart_disease", "심장병")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        draw_rate_chart(
            hyper_result,
            "고혈압 여부에 따른 뇌졸중 비율"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        draw_rate_chart(
            heart_result,
            "심장병 여부에 따른 뇌졸중 비율"
        ),
        use_container_width=True
    )


detail_table = pd.concat([
    hyper_result.assign(질병="고혈압"),
    heart_result.assign(질병="심장병")
], ignore_index=True)

detail_table = detail_table[
    ["질병", "구분", "전체_인원", "뇌졸중_인원", "뇌졸중_비율"]
].rename(columns={
    "전체_인원": "전체 인원",
    "뇌졸중_인원": "뇌졸중 인원",
    "뇌졸중_비율": "뇌졸중 비율 (%)"
})

detail_table["뇌졸중 비율 (%)"] = (
    detail_table["뇌졸중 비율 (%)"].round(2)
)

st.subheader("질병 여부별 상세 통계")
st.dataframe(detail_table, use_container_width=True, hide_index=True)


st.divider()


# ==================================================
# 4. BMI 결측자 비교
# ==================================================

st.header("4. BMI 빈 값이 있는 사람들의 뇌졸중 비율")

valid_stroke_df = df[df["stroke"].isin([0, 1])].copy()

missing_bmi_df = valid_stroke_df[
    valid_stroke_df["bmi"].isna()
].copy()


def get_statistics(data, label):
    count = len(data)
    stroke_count = int((data["stroke"] == 1).sum())

    rate = stroke_count / count * 100 if count else 0

    return {
        "구분": label,
        "전체 인원": count,
        "뇌졸중 인원": stroke_count,
        "뇌졸중 비율 (%)": round(rate, 2)
    }


bmi_table = pd.DataFrame([
    get_statistics(valid_stroke_df, "전체 사람"),
    get_statistics(missing_bmi_df, "BMI 빈 값이 있는 사람")
])

st.dataframe(bmi_table, use_container_width=True, hide_index=True)

st.caption("뇌졸중 여부가 확인된 사람을 기준으로 비율을 계산합니다.")


st.divider()


# ==================================================
# 5. 흡연 상태별 사람 수
# ==================================================

st.header("5. 흡연 상태별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True
)

st.divider()
st.caption("🔎 데이터 탐색 | 뇌졸중 예측 실습실")
