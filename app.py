
import torch
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

def main():
    st.header("Interact with PDF ")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()


        st.success("PDF loaded successfully!")


        st.subheader("PDF Content")
        st.text_area("Text", text, height=100)

     
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

      
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            result = qa_pipeline(context=text, question=query)
            answer = result["answer"]
            st.write("Answer:", answer)

if __name__ == '__main__':
    main()
