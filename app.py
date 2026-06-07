import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os
from PIL import Image

# ==================== 页面设置（必须最先调用） ====================
st.set_page_config(page_title="矿工职业健康评价系统", layout="wide")

# ==================== 中文字体修复（支持 μ、下标等） ====================
# 尝试加载项目目录下的 SimHei.ttf 字体文件
font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
if os.path.exists(font_path):
    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
        st.success("中文字体加载成功（SimHei）")
    except Exception as e:
        st.warning(f"字体加载失败：{e}，使用系统默认中文字体")
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
else:
    # 若字体文件不存在，则使用系统常见中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    st.info("未找到 SimHei.ttf 字体文件，图表可能无法完整显示特殊符号")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'

# ==================== Streamlit 页面字体样式（CSS 回退链） ====================
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC',
                     'SimHei', 'Arial Unicode MS', 'DejaVu Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, label,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    [data-testid="stMetricValue"],
    input, textarea, select {
        font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC',
                     'SimHei', 'Arial Unicode MS', 'DejaVu Sans', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 尝试加载图片（若无图片则跳过）
def load_image(path):
    if os.path.exists(path):
        try:
            return Image.open(path)
        except:
            return None
    return None

col1_img, col2_img, col3_img = st.columns(3)
with col1_img:
    img = load_image("images/coal_mine.jpg")
    if img: st.image(img, caption="煤矿井下作业", use_container_width=True)
with col2_img:
    img = load_image("images/dust_control.jpg")
    if img: st.image(img, caption="粉尘防护措施", use_container_width=True)
with col3_img:
    img = load_image("images/health_check.jpg")
    if img: st.image(img, caption="职业健康检查", use_container_width=True)

st.title("⛑️ 矿山环境矿工职业健康评价系统")
st.markdown("### 输入矿工的职业暴露信息，预测尘肺病分期（0期、I期、II期、III期）")

DATA_FILE = "粉尘数据.xlsx"
MODEL_FILE = "stacking_model.pkl"
SCALER_FILE = "scaler.pkl"
LE_FILE = "le.pkl"

# 训练时使用的原始列名（内部使用）
RAW_COLUMNS = [
    '工序', '确诊年龄', '接尘时间', '粉尘浓度',
    '<2um', '2-5um', '5-10um', '>10um', '游离SiO2', '尘肺病阶段'
]
FEATURE_COLS = RAW_COLUMNS[:-1]

# 页面控件用（普通文本）
DISPLAY_COLS_WIDGET = [
    '工序', '确诊年龄', '接尘时间', '粉尘浓度',
    '<2 um', '2-5 um', '5-10 um', '>10 um', '游离SiO₂'
]

# 图表用（支持数学公式，例如 μ 和下标）
DISPLAY_COLS_PLOT = [
    '工序', '确诊年龄', '接尘时间', '粉尘浓度',
    r'<2 $\mu$m', r'2-5 $\mu$m', r'5-10 $\mu$m', r'>10 $\mu$m', r'游离SiO$_2$'
]

@st.cache_resource
def train_and_save_model():
    df_raw = pd.read_excel(DATA_FILE, sheet_name="Sheet1", header=None, skiprows=2)
    df_raw.columns = RAW_COLUMNS
    for col in df_raw.columns[1:]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw.dropna(inplace=True)

    X = df_raw[FEATURE_COLS].copy()
    y = df_raw['尘肺病阶段'] - 1

    le = LabelEncoder()
    X['工序'] = le.fit_transform(X['工序'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        eval_metric='mlogloss'
    )
    lr = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        stack_method='auto'
    )
    stacking_clf.fit(X_train, y_train)

    joblib.dump(stacking_clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, LE_FILE)
    return stacking_clf, scaler, le

if not os.path.exists(MODEL_FILE):
    with st.spinner("首次运行，正在训练模型，请稍候..."):
        model, scaler, le = train_and_save_model()
    st.success("模型训练完成！")
else:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    le = joblib.load(LE_FILE)

# 计算 Stacking 模型的特征重要性（基模型等权重平均）
rf_imp = model.named_estimators_['rf'].feature_importances_
xgb_imp = model.named_estimators_['xgb'].feature_importances_
lr_coef_abs = np.mean(np.abs(model.named_estimators_['lr'].coef_), axis=0)
stacking_importance = (rf_imp + xgb_imp + lr_coef_abs) / 3

# ==================== 输入表单 ====================
col1, col2 = st.columns(2)

with col1:
    process = st.selectbox("工序", ["落煤", "割煤", "打眼", "放炮", "运转", "检修", "多工序"])
    age = st.number_input("确诊年龄（岁）", min_value=20, max_value=80, value=45, step=1)
    exposure = st.number_input("接尘时间（年）", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
    dust = st.number_input("粉尘浓度（mg/m³）", min_value=0.0, max_value=300.0, value=30.0, step=1.0)
    sio2 = st.number_input("游离SiO₂含量（%）", min_value=0.0, max_value=25.0, value=10.0, step=0.5)

with col2:
    d_lt2 = st.number_input("<2 um 分散度（%）", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    d_2_5 = st.number_input("2-5 um 分散度（%）", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    d_5_10 = st.number_input("5-10 um 分散度（%）", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    d_gt10 = st.number_input(">10 um 分散度（%）", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

process_map = {"落煤": 0, "割煤": 1, "打眼": 2, "放炮": 3, "运转": 4, "检修": 5, "多工序": 6}
process_enc = process_map[process]

input_df = pd.DataFrame(
    [[process_enc, age, exposure, dust, d_lt2, d_2_5, d_5_10, d_gt10, sio2]],
    columns=FEATURE_COLS
)
features_scaled = scaler.transform(input_df)

if st.button("🔍 开始预测", type="primary"):
    proba = model.predict_proba(features_scaled)[0]
    pred_class = int(np.argmax(proba))
    stages = ["0期", "I期", "II期", "III期"]
    pred_stage = stages[pred_class]
    pred_prob = float(proba[pred_class]) * 100

    st.markdown("---")
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.metric("预测分期", pred_stage, delta=f"概率 {pred_prob:.1f}%")
    with col_res2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
        bars = ax.bar(stages, proba, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("概率")
        ax.set_title("各尘肺病分期预测概率")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.2f}", ha='center', fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 特征重要性图
    st.subheader("📊 特征对预测的贡献度（基于 Stacking 集成模型）")
    sorted_idx = np.argsort(stacking_importance)[::-1]
    sorted_names = [DISPLAY_COLS_WIDGET[i] for i in sorted_idx]
    sorted_vals = stacking_importance[sorted_idx]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(sorted_names, sorted_vals, color='#2C5F8A')
    ax2.set_xlabel("重要性")
    ax2.set_title("Stacking 模型特征重要性（基模型加权平均）")
    max_val = max(sorted_vals) * 1.15
    ax2.set_xlim(0, max_val)
    for i, (name, val) in enumerate(zip(sorted_names, sorted_vals)):
        ax2.text(val + 0.002, i, f"{val:.3f}", va='center', fontsize=9)
    st.pyplot(fig2)

    # 健康建议
    st.markdown("### 💡 健康管理建议")
    if pred_stage == "0期":
        st.info("✅ 目前未发现尘肺病征象，建议每年常规职业健康体检，继续做好粉尘防护。")
    elif pred_stage == "I期":
        st.warning("⚠️ 提示：初步诊断为I期尘肺病，建议缩短体检周期至半年一次，加强个体防护，必要时考虑调岗。")
    elif pred_stage == "II期":
        st.error("🆘 提示：初步诊断为II期尘肺病，应尽快调离粉尘岗位，接受规范化治疗和康复训练。")
    else:
        st.error("🚨 提示：初步诊断为III期尘肺病，立即脱离粉尘环境，住院治疗并申请工伤保险。")