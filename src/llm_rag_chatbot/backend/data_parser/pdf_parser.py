import streamlit as st
from unstructured.partition.pdf import partition_pdf

def pdf_parser(input_file):
    try:
        raw_data = partition_pdf(
            filename=input_file,
            strategy='hi_res',
            extract_images_in_pdf=False,
            extract_image_block_to_payload=False,
            extract_image_block_output_dir="./data/images"
        )
        return raw_data
    except Exception as e:
        st.error(f"Failed to extract PDF files: {e}")
        return None

