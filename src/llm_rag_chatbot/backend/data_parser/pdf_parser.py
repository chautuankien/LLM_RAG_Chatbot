import streamlit as st
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from unstructured_inference.models.base import get_model

def pdf_parser(input_file_path: str) -> list[Element]:
    try:
        model = get_model("yolox")
        raw_data = partition_pdf(
            filename=input_file_path,
            strategy='hi_res', # mandatory to infer tables
            hi_res_model_name="detectron2_onnx",
            infer_table_structure=True, #  extract tables
            extract_images_in_pdf=False,
            extract_image_block_types=["Image"], # Add 'Table' to list to extract image of tables
            extract_image_block_to_payload=False, # if true, will extract base64 for API usage
            extract_image_block_output_dir="./data/images", # if None, images and tables will saved in base64     
        )
        return raw_data
    except Exception as e:
        print(e)
        # st.error(f"Failed to extract PDF files: {e}")
        return None


if __name__ == "__main__":
    file = "src/llm_rag_chatbot/backend/data_parser/data_samples/Video_Swin_Transformer.pdf"
    raw_data = pdf_parser(file)
    print(raw_data)