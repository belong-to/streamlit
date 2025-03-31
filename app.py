import streamlit as st
from pathlib import Path
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from shared import upload_records
import pandas as pd
import mysql.connector

# 配置数据库连接信息
config_db = {
    'user': 'root',
    'password': '123456',
    'host': '127.0.0.1',
    'database': 'defect',
    'raise_on_warnings': True
}

# 创建数据库连接
connection = mysql.connector.connect(**config_db)


# 将上传记录写入数据库
def write_upload_records_to_db(records, connection):
    cursor = connection.cursor()
    for record in records:
        query = "INSERT INTO upload_records (file_name, file_type, uploaded_at,result_image) VALUES (%s,%s, %s, %s)"
        print(f"Storing result_image with value: {record['result_image']}")
        cursor.execute(query, (record['file_name'], record['file_type'], record['uploaded_at'], record['result_image']))
    connection.commit()
    cursor.close()


# 设置网页标题和布局
st.set_page_config(page_title="工业零件检测", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <h1 style='text-align: center;'>缺陷发现者</h1>
    <h1 style='text-align: right; font-size: 24px; color: #333; font-weight: bold; margin-bottom: 20px;'>
       ——基于YOLOv8的工业零件检测
    </h1>
""", unsafe_allow_html=True)

# 侧边栏：模型配置
st.sidebar.header("模型配置")
task_options = ["目标检测", "实例分割", "图像分类"]
if 'task_type' not in st.session_state:
    st.session_state['task_type'] = task_options[0]

task_type = st.sidebar.selectbox(
    "任务选择",
    task_options,
    key="task_type"
)

# 使用config模块中的配置
DETECTION_MODEL_LIST = config.DETECTION_MODEL_LIST
INSTANCE_SEGMENTATION_MODEL_LIST = config.INSTANCE_SEGMENTATION_MODEL_LIST
CLASSIFICATION_MODEL_LIST = config.CLASSIFICATION_MODEL_LIST

model_path = ""
if task_type == "目标检测":
    model_type = st.sidebar.selectbox(
        "模型选择",
        DETECTION_MODEL_LIST,
        key="model_type_selectbox"
    )
    model_path = Path(config.DETECTION_MODEL_DIR, model_type)
elif task_type == "实例分割":
    model_type = st.sidebar.selectbox(
        "模型选择",
        INSTANCE_SEGMENTATION_MODEL_LIST,
        key="model_type_selectbox",
        index=1  # 默认选择 best-2.pt
    )
    model_path = Path(config.INSTANCE_SEGMENTATION_MODEL_DIR, model_type)
elif task_type == "图像分类":
    model_type = st.sidebar.selectbox(
        "模型选择",
        CLASSIFICATION_MODEL_LIST,
        key="model_type_selectbox",
        index=2  # 默认选择 best-3.pt
    )
    model_path = Path(config.CLASSIFICATION_MODEL_DIR, model_type)
else:
    st.error("目前仅实现‘目标检测’、‘实例分割’和‘图像分类’功能")

confidence = float(st.sidebar.slider("选择模型置信度", 30, 100, 30)) / 100

# 加载模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# 图像/视频配置
st.sidebar.header("图像/视频配置")
source_selectbox = st.sidebar.selectbox(
    "选择上传类型",
    config.SOURCES_LIST,
    key="source_selectbox"
)
save_path = st.sidebar.text_input("输入保存结果的文件夹路径", "请输入路径", key="save_path_input")

# 根据选择的上传类型调用相应的函数
if source_selectbox == config.SOURCES_LIST[0]:  # Image
    infer_uploaded_image(confidence, model, save_path, task_type)
elif source_selectbox == config.SOURCES_LIST[1]:  # Video
    infer_uploaded_video(confidence, model, save_path, task_type)
elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
    infer_uploaded_webcam(confidence, model, save_path, task_type)
else:
    st.error("目前只支持‘图片’、‘视频’和‘摄像头’类型的上传")

# 显示上传记录
st.subheader("上传记录")
if upload_records:
    # 创建 DataFrame
    upload_df = pd.DataFrame(upload_records)
    # 重置索引，并且不把旧索引添加到 DataFrame 中
    upload_df = upload_df.iloc[:, :4]
    upload_df.columns = ['文件名', '文件类型', '上传时间', '结果路径']
    upload_df = upload_df.reset_index(drop=True)
    # 显示 DataFrame
    st.table(upload_df)

    # 将上传记录写入数据库
    write_upload_records_to_db(upload_records, connection)
else:
    st.write("没有上传记录。")
