# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

import json

# Setting page layout
st.set_page_config(
    page_title="Visual Float Object Detection",
    page_icon="smile",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("浮点智能目标检测演示_web端")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_select = st.sidebar.selectbox('model', settings.DETECTION_LIST)
    # model_path = Path(settings.DETECTION_MODEL)
    model_path = Path(settings.MODEL_DIR / model_select)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                num_bbox = len(res[0].boxes.cls)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                # objects = []
                # data_dec = json.loads(res)
                # for obj_data in data_dec:
                #     # 提取物体信息
                #     obj_index = obj_data['index']  # 物体序号
                #     obj_class = obj_data['class']  # 类别
                #     obj_confidence = obj_data['confidence']  # 置信度
                #     obj_coordinates = obj_data['coordinates']  # 坐标
                #     obj_dec_info = {
                #         '序号': obj_index,
                #         '类别': obj_class,
                #         '置信度': obj_confidence,
                #         '坐标': obj_coordinates
                #     }
                #     objects.append(obj_dec_info)

                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        st.write(num_bbox)
                        for box in boxes:
                            # st.write(box.data)
                            st.write(box.cls, box.conf, box.xyxy)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
