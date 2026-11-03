import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

st.set_page_config(
    page_title=f"분류 모델 | {APP_TITLE}",
    page_icon="🧪",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def accuracy_card(title, test_accuracy, train_accuracy):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.metric("테스트 데이터 정확도", f"{test_accuracy * 100:.2f}%")
        st.caption(
            f"훈련 데이터 정확도: {train_accuracy * 100:.2f}%  ·  "
            f"테스트 데이터 정확도: {test_accuracy * 100:.2f}%"
        )


def tree_dot_graph(model, feature_names):
    """학습된 결정 트리를 훈련 자료 수와 양성 비율을 포함한 DOT로 변환합니다."""
    tree = model.tree_
    lines = [
        "digraph DecisionTree {",
        '  graph [rankdir=TB, nodesep="0.35", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10];',
        '  edge [fontname="sans-serif", fontsize=10];',
    ]

    def walk(node_id):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        values = tree.value[node_id][0]
        total = int(tree.n_node_samples[node_id])
        positive = int(values[list(model.classes_).index(1)]) if 1 in model.classes_ else 0
        ratio = positive / total if total else 0.0

        is_leaf = left == right
        if is_leaf:
            prediction = int(model.classes_[int(np.argmax(values))])
            answer = "뇌졸중" if prediction == 1 else "아님"
            fill = "#f7c9c9" if prediction == 1 else "#cce8d2"
            label = (
                f"답: {answer}\\n"
                f"훈련용 사람: {total}명\\n"
                f"실제 뇌졸중: {positive}명\\n"
                f"뇌졸중 비율: {ratio * 100:.1f}%"
            )
        else:
            feature_index = tree.feature[node_id]
            feature_label = FEATURE_LABELS[feature_names[feature_index]]
            threshold = tree.threshold[node_id]
            fill = "#e8eef7"
            label = (
                f"{feature_label} ≤ {threshold:.3g} ?\\n"
                f"훈련용 사람: {total}명\\n"
                f"실제 뇌졸중: {positive}명\\n"
                f"뇌졸중 비율: {ratio * 100:.1f}%"
            )

        safe_label = html.escape(label, quote=True)
        lines.append(f'  n{node_id} [label="{safe_label}", fillcolor="{fill}"];')

        if not is_leaf:
            walk(left)
            walk(right)
            lines.append(f'  n{node_id} -> n{left} [label="예"];')
            lines.append(f'  n{node_id} -> n{right} [label="아니요"];')

    walk(0)
    lines.append("}")
    return "\n".join(lines)


st.title("🧪 분류 모델")
st.caption(
    "뇌졸중(stroke=1)을 양성, 뇌졸중이 아님(stroke=0)을 음성으로 두고 "
    "두 가지 분류 모델을 학습하고 비교합니다."
)

try:
    data = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

required = ["id", "stroke"] + ALL_FEATURES
missing_columns = [column for column in required if column not in data.columns]
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

# 번호(id) 순으로 정렬하고, 10명 단위 묶음마다 앞 3명을 테스트용으로 고정합니다.
working = data.copy()
working["_id_sort"] = pd.to_numeric(working["id"], errors="coerce")
if working["_id_sort"].notna().all():
    working = working.sort_values("_id_sort", kind="stable")
else:
    working = working.sort_values("id", kind="stable")
working = working.reset_index(drop=True)

# 각 10명 묶음의 0, 1, 2번째 행은 테스트, 나머지는 훈련 자료입니다.
test_mask = (np.arange(len(working)) % 10) < 3
train_raw = working.loc[~test_mask].copy()
test_raw = working.loc[test_mask].copy()

# 레이블이 없는 행은 분류 학습 및 평가에서 제외합니다.
train_raw = train_raw[train_raw["stroke"].isin([0, 1])].copy()
test_raw = test_raw[test_raw["stroke"].isin([0, 1])].copy()

if len(test_raw) != 1533:
    st.info(
        f"현재 데이터에서 테스트용으로 분리된 사람은 {len(test_raw):,}명입니다. "
        "전체 데이터가 안내된 원본과 다르면 1,533명과 달라질 수 있습니다."
    )

if train_raw.empty or test_raw.empty:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()

# 선택한 입력 속성을 숫자형으로 변환합니다.
for column in selected_features:
    train_raw[column] = pd.to_numeric(train_raw[column], errors="coerce")
    test_raw[column] = pd.to_numeric(test_raw[column], errors="coerce")

# BMI를 선택한 경우에만 훈련용 BMI 중앙값으로 훈련/테스트 결측치를 채웁니다.
if "bmi" in selected_features:
    bmi_median = train_raw["bmi"].median()
    if pd.isna(bmi_median):
        st.error("훈련용 데이터에서 체질량지수의 중앙값을 계산할 수 없습니다.")
        st.stop()
    train_raw["bmi"] = train_raw["bmi"].fillna(bmi_median)
    test_raw["bmi"] = test_raw["bmi"].fillna(bmi_median)

# BMI 이외 선택 속성의 결측은 임의 대체하지 않고 해당 행을 제외합니다.
other_features = [feature for feature in selected_features if feature != "bmi"]
if other_features:
    train_raw = train_raw.dropna(subset=other_features)
    test_raw = test_raw.dropna(subset=other_features)

if train_raw.empty or test_raw.empty:
    st.error("선택한 속성의 결측치를 제외한 뒤 훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()

X_train = train_raw[selected_features].astype(float)
X_test = test_raw[selected_features].astype(float)
y_train = train_raw["stroke"].astype(int)
y_test = test_raw["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error("훈련용 데이터에 뇌졸중 있음/없음 두 종류가 모두 있어야 모델을 학습할 수 있습니다.")
    st.stop()

# 크기 맞추기(표준화)는 훈련용 자료에만 fit하고 테스트에는 transform만 적용합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 확률 모델: 로지스틱 회귀
logistic_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
logistic_model.fit(X_train_scaled, y_train)

# 질문 모델: 질문은 최대 3번, 마지막 마디의 최소 사람 수는 5명
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
tree_model.fit(X_train, y_train)

# 아무 입력도 보지 않고 훈련용에서 더 많은 쪽만 답하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())
baseline_train_pred = np.full(len(y_train), majority_class)
baseline_test_pred = np.full(len(y_test), majority_class)

models = [
    (
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_model.predict(X_train_scaled),
        logistic_model.predict(X_test_scaled),
    ),
    (
        "의사결정트리(질문으로 답하는 모델)",
        tree_model.predict(X_train),
        tree_model.predict(X_test),
    ),
    (
        "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        baseline_train_pred,
        baseline_test_pred,
    ),
]

st.subheader("2. 모델 정확도 비교")
accuracy_columns = st.columns(3)
for column, (name, train_pred, test_pred) in zip(accuracy_columns, models):
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    with column:
        accuracy_card(name, test_accuracy, train_accuracy)

st.caption(
    f"훈련용 {len(X_train):,}명 · 테스트용 {len(X_test):,}명 · "
    f"양성(뇌졸중) 테스트 사례 {int((y_test == 1).sum()):,}명"
)


# ------------------------------------------------------------
# 두 속성 평면: 로지스틱 회귀 0.5 경계와 트리 영역
# ------------------------------------------------------------
st.subheader("3. 두 속성으로 보는 분류 결과")
axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda value: FEATURE_LABELS[value],
    )
with axis_col2:
    y_default_index = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=y_default_index,
        format_func=lambda value: FEATURE_LABELS[value],
    )

if x_feature == y_feature:
    st.warning("가로축과 세로축은 서로 다른 속성을 선택해 주세요.")
    st.stop()

# 축으로 사용하지 않는 속성은 테스트 데이터의 중앙값으로 고정합니다.
fixed_values = {}
for feature in selected_features:
    if feature not in (x_feature, y_feature):
        fixed_values[feature] = float(X_test[feature].median())

fixed_description = " · ".join(
    f"{FEATURE_LABELS[feature]}: {value:.3g}"
    for feature, value in fixed_values.items()
)
if fixed_description:
    st.write(f"나머지 속성은 테스트 데이터 중앙값으로 고정했습니다: {fixed_description}")
else:
    st.write("선택한 두 속성이 모두 그림의 축에 사용되므로 고정한 나머지 속성은 없습니다.")

# 관측값 범위에 약간의 여유를 둔 격자를 만듭니다.
x_values = X_test[x_feature].to_numpy(dtype=float)
y_values = X_test[y_feature].to_numpy(dtype=float)
x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
x_pad = (x_max - x_min) * 0.06 if x_max > x_min else 1.0
y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 1.0
x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 180)
y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 180)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_frame = pd.DataFrame(
    {
        feature: np.full(xx.size, fixed_values.get(feature, 0.0))
        for feature in selected_features
    }
)
grid_frame[x_feature] = xx.ravel()
grid_frame[y_feature] = yy.ravel()

# 두 모델 모두 각 격자점에서 나머지 속성은 위에서 안내한 고정값으로 계산합니다.
grid_tree = tree_model.predict(grid_frame[selected_features]).reshape(xx.shape)
grid_scaled = scaler.transform(grid_frame[selected_features])
grid_probability = logistic_model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

fig = go.Figure()

# 트리 분류 영역을 옅은 색으로 표시합니다.
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=grid_tree,
        showscale=False,
        contours=dict(start=0, end=1, size=1, coloring="fill", showlines=False),
        colorscale=[
            [0.0, "rgba(80, 150, 210, 0.12)"],
            [0.499, "rgba(80, 150, 210, 0.12)"],
            [0.5, "rgba(230, 100, 100, 0.12)"],
            [1.0, "rgba(230, 100, 100, 0.12)"],
        ],
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

# 로지스틱 회귀의 양성 확률 0.5 경계
fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=grid_probability,
        showscale=False,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="lines",
            showlabels=False,
        ),
        line=dict(color="#222222", width=3, dash="dash"),
        hoverinfo="skip",
        name="로지스틱 회귀 0.5 경계",
    )
)

# 테스트 자료의 실제 레이블을 점 색상으로 구분합니다.
for label_value, label_name, color in [
    (0, "실제 뇌졸중 아님(0)", "#2878B5"),
    (1, "실제 뇌졸중(1)", "#D62728"),
]:
    subset = test_raw[test_raw["stroke"] == label_value]
    fig.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=label_name,
            marker=dict(color=color, size=7, opacity=0.72, line=dict(width=0.4, color="white")),
            customdata=subset[["id"]].astype(str).to_numpy(),
            hovertemplate=(
                "번호: %{customdata[0]}<br>"
                + f"{FEATURE_LABELS[x_feature]}: "
                + "%{x:.3g}<br>"
                + f"{FEATURE_LABELS[y_feature]}: "
                + "%{y:.3g}<extra>%{fullData.name}</extra>"
            ),
        )
    )

fig.update_layout(
    title="테스트 데이터의 실제 결과와 두 모델의 분류 경계",
    xaxis_title=FEATURE_LABELS[x_feature],
    yaxis_title=FEATURE_LABELS[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650,
)
st.plotly_chart(fig, use_container_width=True)

# 0.5 경계가 현재 표시 범위에 있는지 확인합니다.
has_boundary = bool(
    np.nanmin(grid_probability) <= 0.5 <= np.nanmax(grid_probability)
)
if has_boundary:
    st.write("로지스틱 회귀의 0.5 경계선은 그림 안에 표시되어 있습니다.")
else:
    st.write(
        "로지스틱 회귀의 0.5 경계선은 현재 그림 범위에 나타나지 않습니다. "
        "표시 범위 안에서 확률이 모두 0.5보다 크거나 모두 작습니다."
    )


# ------------------------------------------------------------
# 결정 트리의 질문 흐름
# ------------------------------------------------------------
st.subheader("4. 의사결정트리가 던진 질문")
st.graphviz_chart(tree_dot_graph(tree_model, selected_features), use_container_width=True)

tree_structure = tree_model.tree_
leaf_nodes = [
    node_id
    for node_id in range(tree_structure.node_count)
    if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]
]
negative_leaf_count = 0
for node_id in leaf_nodes:
    values = tree_structure.value[node_id][0]
    prediction = int(tree_model.classes_[int(np.argmax(values))])
    if prediction == 0:
        negative_leaf_count += 1

asked_features = sorted(
    {
        selected_features[feature_index]
        for feature_index in tree_structure.feature
        if feature_index >= 0
    },
    key=lambda feature: selected_features.index(feature),
)
asked_feature_names = ", ".join(FEATURE_LABELS[feature] for feature in asked_features)
if not asked_feature_names:
    asked_feature_names = "없음"

st.write(
    f"답을 내는 마디는 모두 **{len(leaf_nodes)}칸**이며, "
    f"그중 **{negative_leaf_count}칸**이 ‘뇌졸중 아님(0)’이라고 답합니다."
)
st.write(f"이 나무가 실제로 물은 속성: **{asked_feature_names}**")
st.caption(
    "가지의 ‘예’는 질문의 조건(속성 ≤ 기준값)에 해당하고, "
    "‘아니요’는 그보다 큰 경우입니다. 마지막 마디의 색은 예측 결과를 나타냅니다."
)
