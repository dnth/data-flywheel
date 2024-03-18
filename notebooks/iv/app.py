import os
import streamlit as st
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from streamlit_shortcuts import add_keyboard_shortcuts
from PIL import Image


st.set_page_config(layout="wide")

def setup_shortcuts():
    add_keyboard_shortcuts({
        'a': "Previous image",
        'd': "Next image",
        'w': "Save",
        's': "Remove XML file"
    })

def setup_session_state(idm):
    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_exist_annotation_files('relabel_list.txt')
        st.session_state["image_index"] = 0
    else:
        idm.set_annotation_files(st.session_state["files"])

def refresh():
    st.session_state["files"] = idm.get_exist_annotation_files('relabel_list.txt')
    st.session_state["image_index"] = 0

def change_image(delta):
    image_index = st.session_state["image_index"]
    if 0 <= image_index + delta < len(st.session_state["files"]):
        st.session_state["image_index"] += delta
    else:
        st.warning('This is the first/last image.')

def remove_xml_file(xml_dir):
    xml_file_name = st.session_state["files"][st.session_state["image_index"]]
    xml_file_path = os.path.join(xml_dir, xml_file_name)
    os.remove(xml_file_path)
    st.warning(f"Removed XML file: {xml_file_name}")

def go_to_image():
    file_index = st.session_state["files"].index(st.session_state["file"])
    st.session_state["image_index"] = file_index

def save_annotation(im, xml_file_name):
    im.save_annotation()
    st.success(f"Annotation saved to {xml_file_name}")

def calculate_progress():
    current_image_index = st.session_state["image_index"] + 1
    total_files = len(st.session_state["files"])
    progress = (current_image_index / total_files) if total_files > 0 else 0
    return progress

def display_sidebar(xml_dir):
    n_files = len(st.session_state["files"])
    current_image_index = st.session_state["image_index"] + 1
    st.sidebar.write(f"Image {current_image_index} of {n_files}")

    progress = calculate_progress()
    
    st.sidebar.progress(progress)
    st.sidebar.write(f"Progress: {progress*100:.2f}%")
    
    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=change_image, args=(-1,))
    with col2:
        st.button(label="Next image", on_click=change_image, args=(1,))
    
    st.sidebar.button(label="Refresh", on_click=refresh)
    st.sidebar.button(label="Remove XML file", on_click=remove_xml_file, args=(xml_dir,))


def display_annotation(im, labels):
    rects = st_img_label(im.resized_img, box_color="limegreen", rects=im.resized_rects)
    
    if rects:
        preview_imgs = im.init_annotation(rects)
        
        # Add a slider to control the preview image width
        prev_img_width = st.sidebar.slider("Preview Image Width", min_value=100, max_value=500, value=300, step=50)
        
        for i, prev_img in enumerate(preview_imgs):
            # Get the original width and height of the preview image
            original_width, original_height = prev_img[0].size
            
            # Calculate the aspect ratio of the preview image
            aspect_ratio = original_width / original_height
            
            # Calculate the new height based on the desired width and aspect ratio
            new_height = int(prev_img_width / aspect_ratio)
            
            # Resize the image based on the new width and calculated height
            resized_img = prev_img[0].resize((prev_img_width, new_height))
            
            col1, col2 = st.columns(2)
            with col1:
                col1.image(resized_img)
            with col2:
                default_index = labels.index(prev_img[1]) if prev_img[1] else 0
                select_label = col2.selectbox("Label", labels, key=f"label_{i}", index=default_index)
                im.set_annotation(i, select_label)


def run(xml_dir, img_dir, labels):
    st.set_option("deprecation.showfileUploaderEncoding", False)
    idm = ImageDirManager(img_dir, xml_dir)
    setup_session_state(idm)
    setup_shortcuts()
    
    xml_file_name = st.session_state["files"][st.session_state["image_index"]]
    img_file_name = os.path.splitext(xml_file_name)[0] + ".jpg"
    img_path = os.path.join(img_dir, img_file_name)
    xml_path = os.path.join(xml_dir, xml_file_name)
    
    if not os.path.exists(img_path):
        st.warning(f"Image file not found: {img_file_name}")
        return
    
    im = ImageManager(img_path, xml_path)
    im.resized_img = im.resizing_img()
    im.resized_rects = im.get_resized_rects()
    
    st.subheader(f"Image filename: {img_file_name}")
    display_sidebar(xml_dir)
    display_annotation(im, labels)
    
    
    st.button(label="Save", on_click=save_annotation, args=(im, xml_file_name))

if __name__ == "__main__":
    custom_labels = ["bowling_ball"]
    run("pascal_voc_annotations_bowling_ball/", '/workspace/yolo_v8_training/oiv7_full/validation/', custom_labels)