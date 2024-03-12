import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from streamlit_shortcuts import add_keyboard_shortcuts

add_keyboard_shortcuts({
    'a': "Previous image",
    'd': "Next image",
    'w': "Save",
    's': "Remove XML file"
})

def run(xml_dir, img_dir, labels):
    st.set_option("deprecation.showfileUploaderEncoding", False)
    idm = ImageDirManager(img_dir, xml_dir)
    
    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_annotation_files(st.session_state["files"])
    
    def refresh():
        st.session_state["files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    
    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')
    
    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')
    
    def remove_xml_file():
        xml_file_name = st.session_state["files"][st.session_state["image_index"]]
        xml_file_path = os.path.join(xml_dir, xml_file_name)
        os.remove(xml_file_path)
        st.warning(f"Removed XML file: {xml_file_name}")
    
    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index
    
    def save_annotation():
        im.save_annotation()
        st.success(f"Annotation saved to {xml_file_name}")
    
    # Sidebar: show status
    n_files = len(st.session_state["files"])
    st.sidebar.write("Total annotate files:", n_files)
    
    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=previous_image)
    with col2:
        st.button(label="Next image", on_click=next_image)
    
    st.sidebar.button(label="Refresh", on_click=refresh)
    st.sidebar.button(label="Remove XML file", on_click=remove_xml_file)
    
    # Main content: display images and annotations
    xml_file_name = st.session_state["files"][st.session_state["image_index"]]
    img_file_name = os.path.splitext(xml_file_name)[0] + ".jpg"
    img_path = os.path.join(img_dir, img_file_name)
    xml_path = os.path.join(xml_dir, xml_file_name)
    
    if not os.path.exists(img_path):
        st.warning(f"Image file not found: {img_file_name}")
        return
    
    im = ImageManager(img_path, xml_path)
    img = im.get_img()
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()
    
    rects = st_img_label(resized_img, box_color="blue", rects=resized_rects)
    
    if rects:
        preview_imgs = im.init_annotation(rects)
        for i, prev_img in enumerate(preview_imgs):
            prev_img[0].thumbnail((300, 300))
            col1, col2 = st.columns(2)
            with col1:
                col1.image(prev_img[0])
            with col2:

                default_index = 0
                if prev_img[1]:
                    default_index = labels.index(prev_img[1])

                select_label = col2.selectbox(
                    "Label",
                    labels,
                    key=f"label_{i}",
                    index=default_index
                )
                im.set_annotation(i, select_label)
    
    st.button(label="Save", on_click=save_annotation)

if __name__ == "__main__":
    custom_labels = ["person"]
    run("pascal_voc_annotations/", '/workspace/yolo_v8_training/oiv7_full/validation/', custom_labels)