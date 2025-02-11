import io
from collections.abc import Sequence
import streamlit as st
from uuid import uuid4

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, NarrativeText, Image, Table

from backend.prompts import IMAGE_SUMMARY_PROMPT, TABLE_SUMMARY_PROMPT

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

@st.cache_data(show_spinner=True)
def pdf_parser(input_data: io.BytesIO| str| None) -> list[Element]|None:
    """
    Parses a PDF file from a given input and extracts its elements.

    Args:
        input_data (io.BytesIO| str| None): The input data to parse. It can be:
            - None: If no input is provided.
            - str: A string representing the file path of the PDF.
            - io.BytesIO: A BytesIO object containing the PDF data.

    Returns:
        list[Element]|None: A list of extracted elements from the PDF if successful, otherwise None.

    Raises:
        Exception: If an error occurs during PDF parsing, an error message is logged and None is returned.
    """
    try:
        if input_data is None:
            return None
        elif isinstance(input_data, str):
            elements: list[Element] = partition_pdf(
                filename=input_data,
                strategy='hi_res', # mandatory to infer tables
                hi_res_model_name="yolox",
                infer_table_structure=True, #  extract tables
                extract_images_in_pdf=True,
                extract_image_block_types=["Image"], # Add 'Table' to list to extract image of tables
                extract_image_block_to_payload=True, # if true, will extract base64 for API usage
                extract_image_block_output_dir=None, # if None, images and tables will saved in base64     
            )
            return elements
        else:
            elements: list[Element] = partition_pdf(
                file=input_data,
                strategy='hi_res', # mandatory to infer tables
                hi_res_model_name="yolox",
                infer_table_structure=True, #  extract tables
                extract_images_in_pdf=True,
                extract_image_block_types=["Image"], # Add 'Table' to list to extract image of tables
                extract_image_block_to_payload=True, # if true, will extract base64 for API usage
                extract_image_block_output_dir=None, # if None, images and tables will saved in base64     
            )
            return elements
    except Exception as e:
        st.error(body=f"Failed to extract PDF files: {e}")
        return None


#####################################
####### Extract Text Data ###########
#####################################
def extract_text_data(elements: list[Element] | None) -> list[dict]:
    """
    Extracts text data from a list of elements and returns it as a list of dictionaries.
    Args:
        elements (Optional[list[Element]]): A list of Element objects or None.
    Returns:
        list[dict]: A list of dictionaries containing extracted text data. Each dictionary has the following keys:
            - "src_path" (str): The source path constructed from the file directory and filename.
            - "data" (str): An empty string (reserved for future use).
            - "description" (str): The text content of the element.
            - "page_number" (Optional[int]): The page number of the element, if available.
            - "file_type" (str): The type of file, which is always "text".
    """
    text_data: list[dict] = []
    if elements is None:
        return text_data
    for element in elements:
        if isinstance(element, NarrativeText):
            file_directory: str = element.metadata.file_directory or ""
            filename: str = element.metadata.filename or ""

            src_path: str = file_directory + '/' + filename
            text_content: str = element.text
            page_number: int|None = element.metadata.page_number

            text_data.append({
                "src_path": src_path,
                "data": "",
                "description": text_content,
                "page_number": page_number,
                "file_type": "text"
            })
            
    return text_data

#####################################
####### Extract Image Data ###########
#####################################
def extract_image_data_with_summary(elements: list[Element] | None, decription_model: BaseLanguageModel) -> list[dict]:
    """
    Extracts image data from a list of elements and generates a summary description for each image using a language model.
    Args:
        elements (list[Element] | None): A list of elements that may contain images. If None, an empty list is returned.
        decription_model (BaseLanguageModel): A language model used to generate descriptions for the images.
    Returns:
        list[dict]: A list of dictionaries, each containing the following keys:
            - "src_path" (str): The source path of the image.
            - "data" (str | None): The base64 encoded image data.
            - "description" (str): The description generated by the language model.
            - "page_number" (int | None): The page number where the image is located.
            - "file_type" (str): The type of the file, which is "image".
    """
    image_data: list[dict] = []
    if elements is None:
        return image_data
    
    # Create ChatPromptTemplate instance
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template=IMAGE_SUMMARY_PROMPT)
 
    for element in elements:
        if isinstance(element, Image):
            file_directory: str = element.metadata.file_directory or ""
            filename: str = element.metadata.filename or ""

            src_path: str = file_directory + '/' + filename
            image_base64: str | None = element.metadata.image_base64
            page_number: int | None = element.metadata.page_number
            chain = prompt_template | decription_model
            description = chain.invoke(input={"image_data": f"{image_base64}"})

            image_data.append({
                "src_path": src_path,
                "data": image_base64,
                "description": description,
                "page_number": page_number,
                "file_type": "image"
            })
    return image_data


#####################################
####### Extract Table Data ###########
#####################################
def extract_table_data_with_summary(elements: list[Element] | None, decription_model: BaseLanguageModel) -> list[dict]:
    """
    Extracts table data from a list of elements and generates a summary description for each table.
    Args:
        elements (list[Element] | None): A list of elements to process. Each element is expected to be an instance of the `Element` class.
        decription_model (BaseLanguageModel): A language model used to generate descriptions for the table data.
    Returns:
        list[dict]: A list of dictionaries containing the extracted table data and their corresponding descriptions. Each dictionary has the following keys:
            - "src_path" (str): The source path of the file containing the table.
            - "data" (str | None): The HTML representation of the table data.
            - "description" (str): The generated description of the table data.
            - "page_number" (int | None): The page number where the table is located.
            - "file_type" (str): The type of file, which is always "table" in this case.
    """
    table_data: list[dict] = []
    if elements is None:
        return table_data
    
    # Create ChatPromptTemplate instance
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template=TABLE_SUMMARY_PROMPT)

    for element in elements:
        if isinstance(element, Table):
            file_directory: str = element.metadata.file_directory or ""
            filename: str = element.metadata.filename or ""

            src_path: str = file_directory + '/' + filename
            text_as_html: str | None = element.metadata.text_as_html
            chain = prompt_template | decription_model
            description = chain.invoke(input={"table_data": f"{text_as_html}"})
            page_number: int | None = element.metadata.page_number

            table_data.append({
                "src_path": src_path,
                "data": text_as_html,
                "description": description,
                "page_number": page_number,
                "file_type": "table"
            })
    return table_data


def convert_to_documents(data: list[dict]) -> list[Document]:
    """
    Converts a list of dictionaries into a list of Document objects.
    Args:
        data (list[dict]): A list of dictionaries where each dictionary contains 
                           the keys 'description', 'src_path', 'page_number', and 'file_type'.
    Returns:
        list[Document]: A list of Document objects with the content and metadata 
                        extracted from the input dictionaries.
    """
    doc_ids: list[str] = [str(uuid4) for _ in range(len(data))]

    documents: list[Document] = [
        Document(
            page_content=doc['description'],
            metadata={"id": doc_ids[i], 
                      "src_path": doc['src_path'], 
                      "page_number": doc['page_number'], 
                      "file_type": doc['file_type']}
        )
        for i, doc in enumerate(iterable=data)
    ]

    return documents

def extract_and_convert_all_data(elements: list[Element] | None, decription_model: BaseLanguageModel) -> Sequence[list[Document]]:
    text_data: list[dict] = extract_text_data(elements=elements)
    # image_data: list[dict] = extract_image_data_with_summary(elements=elements, decription_model=decription_model)
    # table_data: list[dict] = extract_table_data_with_summary(elements=elements, decription_model=decription_model)

    text_docs: list[Document] = convert_to_documents(data=text_data)
    # image_docs: list[Document] = convert_to_documents(data=image_data)
    # table_docs: list[Document] = convert_to_documents(data=table_data)
    image_docs, table_docs = None, None

    return text_docs, image_docs, table_docs

