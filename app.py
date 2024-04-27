import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
#from langchain.embeddings import BedrockEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import warnings
from io import StringIO
import sys
import os
import boto3
import textwrap
import json
import botocore
from botocore.config import Config
import base64
import time 
from io import BytesIO

warnings.filterwarnings('ignore')

# Configura el cliente de AWS Bedrock
boto3_bedrock = boto3.client('bedrock-runtime')

question_answer_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store  # Retorna directamente el vector store 

def get_conversational_chain(vector_store):
    prompt_template = """
    Responda la pregunta lo más detalladamente posible desde el contexto proporcionado,
    asegúrese de proporcionar todos los detalles, si la respuesta no está en el contexto proporcionado,
    simplemente diga 'la respuesta no está disponible en el contexto', no proporcione la respuesta incorrecta.

    Context: \n{context}\n
    Question: \n{question}\n

    Assistant:
    """
    model = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':400})
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),  # Asumiendo que vector_store tiene el método as_retriever
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": prompt})
    return chain

# Mmanejar correctamente el tipo de vector_store que retornas desde get_vector_store.

def user_input(user_question, vector_store_faiss):
    existing_questions = [qa['question'] for qa in st.session_state['question_answer_history']]
    if user_question not in existing_questions:
        chain = get_conversational_chain(vector_store_faiss)
        response = chain({"query": user_question})
        full_answer = response['result']
        st.session_state['question_answer_history'].append({"question": user_question, "answer": full_answer})
        return full_answer
    else:
        for qa in st.session_state['question_answer_history']:
            if qa['question'] == user_question:
                return qa['answer']

# Función para generar y descargar el historial como un archivo .txt
def get_text_download_link(text_to_download, filename, text):
    b64 = base64.b64encode(text_to_download.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Función que crea el historial de texto para descargar
def create_downloadable_history(history):
    text_to_download = '\n\n'.join(f"Pregunta: {qa['question']}\nRespuesta: {qa['answer']}" for qa in history)
    return text_to_download


def main():
    # Configuración inicial de la página
    st.set_page_config(page_title="Asistente Virtual Emssanar", page_icon=":scroll:")
    st.header("⚖️ Realiza tu consulta 📚")

    # Inicialización de la sesión para el historial de preguntas y respuestas
    if 'question_answer_history' not in st.session_state:
        st.session_state['question_answer_history'] = []
    
    #if st.session_state['question_answer_history']:
     #   download_text = create_downloadable_history(st.session_state['question_answer_history'])
      #  st.markdown(get_text_download_link(download_text, 'historial.txt', 'Descargar Historial de Preguntas y Respuestas'), unsafe_allow_html=True)


    # Sidebar para la carga de archivos y procesamiento
    with st.sidebar:
        #html_title = "<h1 style='text-align: lefht; font-size:15px;font-weight: bold;'>⚖️ ASISTENTE IA EMSSANAR EPS </h1>" #font-family:Verdana, sans-serif;
        #st.markdown(html_title, unsafe_allow_html=True)
        imagen = "Logo_Emssanar_EPS.png"
        st.image(imagen)#,align='center')
        st.write("---")

        st.title("📁 Sección de Archivos")
        pdf_docs = st.file_uploader("Por favor cargue sus documentos PDF aquí y después puede dar click en botón Enviar y procesar:", accept_multiple_files=True)

        if st.button("Enviar y procesar"):
            if pdf_docs:  # Verifica si la lista de documentos cargados no está vacía
                with st.spinner("Leyendo..."):
                # Asumimos que estas funciones están definidas para procesar los documentos
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store_faiss = get_vector_store(text_chunks)
                    st.session_state['vector_store'] = vector_store_faiss
                    st.success("En línea")
            else:  # Si pdf_docs está vacío, muestra un mensaje de error
                st.error("Por favor, carga al menos un documento PDF para poder ayudarte.")

        
        
        #st.write("---")
        st.image("oie_30144137iJGmGE90.gif")
        #st.write("---")
        st.write()  # 
        texto = "<p style='font-size: 10px;text-align: center;'>| © Emssanar 2024 | \n Creado por: Ing. Adrian Hidalgo.</p>"
        st.write(texto, unsafe_allow_html=True)
    
    # Entrada y procesamiento de preguntas del usuario
    user_question = st.text_input("Ingrese su pregunta sobre la información de los archivos PDF cargados...")

    # Botón para procesar la pregunta
    if st.button("Procesar pregunta"):
        if user_question and 'vector_store' in st.session_state:
            # Inicio de la barra de progreso
            progress_bar = st.progress(0)
            status_message = st.empty()
            status_message.text("Procesando tu pregunta...")

            # Simulación de actualización de la barra de progreso
            for percent_complete in range(100):
                time.sleep(0.1)  # Simular un proceso que tarda tiempo
                progress_bar.progress(percent_complete + 1)

            # Llamada a la función que maneja la lógica de generación de respuestas
            answer = user_input(user_question, st.session_state['vector_store'])

            # Finalización de la barra de progreso y eliminación del mensaje de estado
            progress_bar.empty()
            status_message.empty()
            status_message.empty()
            # Mostrar la respuesta
            st.write("Respuesta:", answer)
        else:
            st.error("Por favor, introduce una pregunta para continuar.")

        if st.session_state['question_answer_history']:
            with st.expander("Descargar Historial de Preguntas y Respuestas"):
                download_text = create_downloadable_history(st.session_state['question_answer_history'])
                st.markdown(get_text_download_link(download_text, 'historial.txt', 'Descargar Historial'), unsafe_allow_html=True)

    # Botón para mostrar/ocultar el historial de preguntas y respuestas
    if st.button('Mostrar/Ocultar Historial'):
        st.session_state['show_history'] = not st.session_state.get('show_history', False)

    if st.session_state.get('show_history', False):
        st.subheader("Historial de Preguntas y Respuestas:")
        for qa in st.session_state['question_answer_history']:
            st.text(f"Pregunta: {qa['question']}\nRespuesta: {qa['answer']}\n")
            
    html_text = """
    <div style="position: fixed; bottom: 0; left: 0; width: 98%; padding: 10px; text-align: right;font-size:15px;">
    Para más información visite: <a href=" https://emssanareps.co/" target="_blank">www.emssanareps.co</a> 
    </div>
    """
    # Mostrar el HTML usando st.markdown
    st.markdown(html_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
