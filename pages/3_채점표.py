import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = list(FEATURE_LABELS.keys())
DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

MODEL_LOGISTIC = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"
MODEL_BASELINE = "훈련용에서 많은 쪽으로만 답하는 모델"


st.set_page_config(
    page_title=f"채점표 | {APP_TITLE}",
    page_icon="🧾",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def split_and_prepare(data, selected_features):
    """분류 모델 페이지와 동일한 순서·분할·결측치 처리 절차를 적용합니다."""
    working = data.copy()
    working["_id_sort"] = pd.to_numeric(working["id"], errors="coerce")
    if working["_id_sort"].notna().all():
        working = working.sort_values("_id_sort", kind="stable")
    else:
        working = working.sort_values("id", kind="stable")
    working = working.reset_index(drop=True)

    # 번호순 정렬 후 10명 묶음의 앞 3명을 테스트용으로 고정
    test_mask = (np.arange(len(working)) % 10) < 3
    train_raw = working.loc[~test_mask].copy()
    test_raw = working.loc[test_mask].copy()

    train_raw = train_raw[train_raw["stroke"].isin([0, 1])].copy()
    test_raw = test_raw[test_raw["stroke"].isin([0, 1])].copy()

    for column in selected_features:
        train_raw[column] = pd.to_numeric(train_raw[column], errors="coerce")
        test_raw[column] = pd.to_numeric(test_raw[column], errors="coerce")

    # BMI를 선택한 경우에만 훈련용 중앙값으로 두 자료의 빈 값을 채움
    bmi_median = None
    if "bmi" in selected_features:
        bmi_median = train_raw["bmi"].median()
        if pd.isna(bmi_median):
            raise ValueError("훈련용 데이터에서 체질량지수의 중앙값을 계산할 수 없습니다.")
        train_raw["bmi"] = train_raw["bmi"].fillna(bmi_median)
        test_raw["bmi"] = test_raw["bmi"].fillna(bmi_median)

    # BMI 이외 선택 속성의 결측 행은 분류 모델 페이지와 동일하게 제외
    other_features = [feature for feature in selected_features if feature != "bmi"]
    if other_features:
        train_raw = train_raw.dropna(subset=other_features)
        test_raw = test_raw.dropna(subset=other_features)

    if train_raw.empty or test_raw.empty:
        raise ValueError("결측치 처리 후 훈련용 또는 테스트용 데이터가 없습니다.")

    X_train = train_raw[selected_features].astype(float)
    X_test = test_raw[selected_features].astype(float)
    y_train = train_raw["stroke"].astype(int)
    y_test = test_raw["stroke"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("훈련용 데이터에 뇌졸중 있음/없음 두 종류가 모두 있어야 합니다.")

    return train_raw, test_raw, X_train, X_test, y_train, y_test, bmi_median


def confusion_counts(y_true, y_pred):
    """양성=1 기준 TP, FP, FN, TN을 계산합니다."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def make_confusion_table(counts):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    return pd.DataFrame(
        [
            {
                "실제 \\ 예측": "실제 뇌졸중(1)",
                "예측 뇌졸중(1)": f"TP: {tp:,}명",
                "예측 아님(0)": f"FN: {fn:,}명",
                "행 합계": f"{tp + fn:,}명",
            },
            {
                "실제 \\ 예측": "실제 아님(0)",
                "예측 뇌졸중(1)": f"FP: {fp:,}명",
                "예측 아님(0)": f"TN: {tn:,}명",
                "행 합계": f"{fp + tn:,}명",
            },
            {
                "실제 \\ 예측": "열 합계",
                "예측 뇌졸중(1)": f"{tp + fp:,}명",
                "예측 아님(0)": f"{fn + tn:,}명",
                "행 합계": f"{tp + fp + fn + tn:,}명",
            },
        ]
    )


def calculate_metrics(counts):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    total = tp + fp + fn + tn
    actual_positive = tp + fn
    predicted_positive = tp + fp

    accuracy = (tp + tn) / total if total else None
    recall = tp / actual_positive if actual_positive else None
    precision = tp / predicted_positive if predicted_positive else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def metric_text(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


def make_metrics_table(all_counts):
    rows = []
    for model_name, counts in all_counts.items():
        metrics = calculate_metrics(counts)
        rows.append(
            {
                "모델": model_name,
                "정확도": metric_text(metrics["정확도"]),
                "재현율": metric_text(metrics["재현율"]),
                "정밀도": metric_text(metrics["정밀도"]),
                "F1": metric_text(metrics["F1"]),
            }
        )
    return pd.DataFrame(rows)


def person_list(data, mask, title):
    selected = data.loc[mask, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]].copy()
    selected["age"] = pd.to_numeric(selected["age"], errors="coerce")
    selected = selected.sort_values("age", ascending=False, na_position="last")
    total_count = len(selected)

    display = selected.head(20).rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )
    st.markdown(f"**{title} — 전체 {total_count:,}명**")
    if total_count == 0:
        st.caption("해당하는 사람이 없습니다.")
    else:
        st.dataframe(display, hide_index=True, use_container_width=True)
        if total_count > 20:
            st.caption(f"나이가 많은 순서로 앞 20명만 표시했습니다. 전체 인원은 {total_count:,}명입니다.")


st.title("🧾 채점표")
st.caption(
    "뇌졸중(stroke=1)을 양성, 뇌졸중이 아님(stroke=0)을 음성으로 평가합니다. "
    "재현율은 실제 뇌졸중인 사람을 얼마나 찾아냈는지, 정밀도는 뇌졸중이라고 한 사람 중 실제 환자가 얼마나 되는지를 나타냅니다."
)

try:
    data = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

required_columns = ["id", "stroke"] + ALL_FEATURES
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()

st.subheader("1. 입력 속성 고르기")
selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda value: FEATURE_LABELS[value],
    help="체질량지수는 선택한 경우에만 훈련용 데이터의 중앙값으로 빈 값을 채웁니다.",
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

try:
    (
        train_raw,
        test_raw,
        X_train,
        X_test,
        y_train,
        y_test,
        bmi_median,
    ) = split_and_prepare(data, selected_features)
except ValueError as error:
    st.error(str(error))
    st.stop()

if len(test_raw) != 1533:
    st.info(
        f"현재 테스트용 데이터는 {len(test_raw):,}명입니다. "
        "안내된 원본 데이터와 행 수가 다르면 1,533명과 달라질 수 있습니다."
    )

# 크기 맞추기는 훈련용 데이터에만 맞추고 테스트용에는 변환만 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 분류 모델 페이지와 동일한 모델 설정
logistic_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
logistic_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_model.fit(X_train, y_train)

# 훈련용에서 다수인 레이블만 답하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

predictions = {
    MODEL_LOGISTIC: logistic_model.predict(X_test_scaled),
    MODEL_TREE: tree_model.predict(X_test),
    MODEL_BASELINE: np.full(len(y_test), majority_class, dtype=int),
}

all_counts = {
    model_name: confusion_counts(y_test, predicted)
    for model_name, predicted in predictions.items()
}

# ------------------------------------------------------------
# 테스트 데이터 요약
# ------------------------------------------------------------
test_positive_count = int((y_test == 1).sum())
st.subheader("2. 테스트 데이터")
st.write(
    f"테스트용 사람 수: **{len(y_test):,}명**　|　"
    f"그중 실제 뇌졸중인 사람 수: **{test_positive_count:,}명**"
)

# ------------------------------------------------------------
# 혼동행렬
# ------------------------------------------------------------
st.subheader("3. 모델별 혼동행렬")
st.caption(
    "행은 실제 결과, 열은 모델의 예측입니다. "
    "TP=실제 뇌졸중을 뇌졸중으로 예측, FN=실제 뇌졸중을 놓침, "
    "FP=아닌 사람을 뇌졸중으로 잘못 예측, TN=아닌 사람을 아니라고 예측."
)

matrix_col1, matrix_col2 = st.columns(2)
for column, model_name in [
    (matrix_col1, MODEL_LOGISTIC),
    (matrix_col2, MODEL_TREE),
]:
    with column:
        st.markdown(f"**{model_name}**")
        st.dataframe(
            make_confusion_table(all_counts[model_name]),
            hide_index=True,
            use_container_width=True,
        )

# 기준 모델 혼동행렬도 지표 비교와 일관되게 확인할 수 있도록 표시
with st.expander(MODEL_BASELINE + "의 혼동행렬 보기"):
    st.dataframe(
        make_confusion_table(all_counts[MODEL_BASELINE]),
        hide_index=True,
        use_container_width=True,
    )

# ------------------------------------------------------------
# 네 가지 평가 지표
# ------------------------------------------------------------
st.subheader("4. 네 가지 평가 지표 비교")
st.dataframe(
    make_metrics_table(all_counts),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "정확도=(TP+TN)/전체, 재현율=TP/(TP+FN), "
    "정밀도=TP/(TP+FP), F1=2×정밀도×재현율/(정밀도+재현율). "
    "정밀도는 뇌졸중 예측이 0명이면 계산할 수 없습니다."
)

# ------------------------------------------------------------
# 놓친 환자와 헛짚은 사람 목록
# ------------------------------------------------------------
st.subheader("5. 놓친 환자와 헛짚은 사람")
selected_model = st.selectbox(
    "목록을 확인할 모델을 고르세요.",
    options=[MODEL_LOGISTIC, MODEL_TREE, MODEL_BASELINE],
)

selected_prediction = predictions[selected_model]
false_negative_mask = (y_test.to_numpy() == 1) & (selected_prediction == 0)
false_positive_mask = (y_test.to_numpy() == 0) & (selected_prediction == 1)

list_col1, list_col2 = st.columns(2)
with list_col1:
    person_list(
        test_raw,
        false_negative_mask,
        "놓친 환자(실제 뇌졸중인데 아님으로 예측)",
    )

with list_col2:
    person_list(
        test_raw,
        false_positive_mask,
        "헛짚은 사람(실제 아님인데 뇌졸중으로 예측)",
    )

fn_count = int(false_negative_mask.sum())
fp_count = int(false_positive_mask.sum())
matrix_fn = all_counts[selected_model]["FN"]
matrix_fp = all_counts[selected_model]["FP"]

st.write(
    f"선택한 모델의 놓친 환자 **{fn_count:,}명 = 혼동행렬 FN {matrix_fn:,}명**, "
    f"헛짚은 사람 **{fp_count:,}명 = 혼동행렬 FP {matrix_fp:,}명**입니다."
)
