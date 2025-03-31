import sqlite3

from shared import upload_records
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import datetime
import os
import io
import time
import mysql.connector
def _display_detected_frames(conf, model, st_frame, image, save_path, task_type):
    """
    Display the detected objects on a video frame using the YOLO model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLO): An instance of the YOLO class containing the YOLO model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :param save_path (str): The path to save the results.
    :param task_type (str): The type of task, either 'detection' or 'segmentation'.
    :return: None
    """
    # Ensure the image is a 3-channel彩色图像
    if image.ndim == 2 or image.shape[2] == 1:  # 灰度图像或单通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # 四通道RGBA图像
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # Resize the image to the standard size expected by the model
    image_resized = cv2.resize(image, (640, 480))

    # Perform object detection or segmentation using the YOLO model
    results = model.predict(image_resized, conf=conf)

    # Convert the results to the correct format for display and saving
    if task_type == 'detection':
        result_image = results[0].plot()
    else:  # segmentation
        result_image = results[0].plot()

    # Convert from BGR to RGB for Streamlit display
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Resize the result image to the fixed output size (750, 500) while maintaining aspect ratio
    h, w = result_image_rgb.shape[:2]
    scale_factor = min(550 / w, 450 / h)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    result_image_resized = cv2.resize(result_image_rgb, (new_w, new_h))

    # Pad the image to ensure it is 750x500
    padded_image = np.full((500, 750, 3), 255, dtype=np.uint8)  # Create a white background
    start_x = (750 - new_w) // 2
    start_y = (500 - new_h) // 2
    padded_image[start_y:start_y+new_h, start_x:start_x+new_w, :] = result_image_resized

    # Display the frame with detections or segmentations in the Streamlit app
    st_frame.image(
        padded_image,  # Directly use RGB image for display
        caption=f'运行结果',
        use_column_width=True
    )
    # 配置数据库连接信息
    config_db = {
        'user': 'root',
        'password': '123456',
        'host': '127.0.0.1',
        'database': 'defect',
        'raise_on_warnings': True
    }
    # If a save path is provided, save the frame with detections or segmentations
    if save_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_type}_frame_{timestamp}.png"
        save_path_full = os.path.join(save_path, filename)
        # Save the padded image in RGB format
        cv2.imwrite(save_path_full, result_image_resized)  # Save in RGB format
        st.write(f"文件保存在: {save_path_full}")
        if upload_records:
                    upload_records[-1]["result_name"] = save_path_full


# 全局模型缓存字典
_model_cache = {}

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection or segmentation model from the specified model_path.
    Uses caching to avoid reloading the same model multiple times.
    Parameters:
        model_path (str): The path to the YOLO model file.
    Returns:
        A YOLO object detection or segmentation model.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    model = YOLO(model_path)
    _model_cache[model_path] = model
    return model

def infer_uploaded_image(conf, model, save_path, task_type):
    """
    Execute inference for uploaded images in batch.
    :param conf: Confidence of YOLO model
    :param model: An instance of the YOLO class containing the YOLO model.
    :param save_path: The path to save the results.
    :param task_type: The type of task, either 'detection' or 'segmentation'.
    :return: None
    """
    source_imgs = st.sidebar.file_uploader(
        "选择图像",
        type=("jpg", "jpeg", "png", 'bmp', 'webp'),
        accept_multiple_files=True,
    )

    if source_imgs:
        for img_info in source_imgs:
            file_type = os.path.splitext(img_info.name)[1][1:].lower()
            upload_records.append({
                "file_name": img_info.name,
                "file_type": file_type,
                "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result_image": save_path
            })

            uploaded_image = Image.open(img_info)
            img_byte_arr = io.BytesIO()
            uploaded_image.save(img_byte_arr, format=file_type.upper() if file_type != 'jpg' else 'JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            image = np.array(Image.open(io.BytesIO(img_byte_arr)))

            st.image(
                img_byte_arr,
                caption=f"上传的图像: {img_info.name}",
                use_column_width=True
            )

            with st.spinner("正在运行..."):
                _display_detected_frames(conf, model, st.empty(), image, save_path, task_type)

def infer_uploaded_video(conf, model, save_path, task_type):
    """
    Execute inference for uploaded video and display the detected objects on the video.
    :param conf: Confidence of YOLO model
    :param model: An instance of the YOLO class containing the YOLO model.
    :param save_path: The path to save the results.
    :param task_type: The type of task, either 'detection' or 'segmentation'.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        "选择视频",
        accept_multiple_files=True
    )

    if source_video:
        for video_file in source_video:
            file_type = os.path.splitext(video_file.name)[1][1:].lower()
            upload_records.append({
                "file_name": video_file.name,
                "file_type": file_type,
                "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result_image": save_path

            })

            st.video(video_file)

            if st.button("开始运行"):
                with st.spinner("运行中..."):
                    try:
                        tfile = tempfile.NamedTemporaryFile()
                        tfile.write(video_file.read())
                        vid_cap = cv2.VideoCapture(tfile.name)
                        st_frame = st.empty()
                        frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
                        delay = int(1000 / frame_rate)

                        start_time = time.time()
                        while True:
                            success, image = vid_cap.read()
                            if not success:
                                break

                            current_time = time.time()
                            if current_time - start_time >= 1.0:
                                _display_detected_frames(conf, model, st_frame, image, save_path, task_type)
                                start_time = current_time

                        vid_cap.release()
                    except Exception as e:
                        st.error(f"Error loading video: {e}")

def infer_uploaded_webcam(conf, model, save_path, task_type):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLO model
    :param model: An instance of the YOLO class containing the YOLO model.
    :param save_path: The path to save the results.
    :param task_type: The type of task, either 'detection' or 'segmentation'.
    :return: None
    """
    try:
        flag = st.button(
            "关闭摄像头"
        )
        vid_cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image, save_path, task_type)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
