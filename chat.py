import streamlit as st
from mistralai import Mistral, UserMessage #libraries importieren

#hier kommen libraries für PDF 
import io
import numpy as np
import PyPDF2
import faiss

col1, col2 = st.columns(2)

api_key = "Gj7JfePqwQ4IZqFm8CbjsDYgTjnjd5a9" #API-Key sollte normalerweisen nicht im Code sein
client = Mistral(api_key=api_key) #API-Key erlaubt ie Kommuniktion mit der Maschine

st.title("Mr. Crocker (Dein KI-Assistent)") #Titel für Webapp

if "messages" not in st.session_state: #Die Seesion-State ist der ChatVerlauf
    st.session_state.messages = [] #Erstellt eine leere Liste für den Chatverlauf, damit keine Fehler wegen einer fehlenden Liste auftauchen
    st.session_state.pdfs = [] #ermöglicht das gleiche für PDF

for message in st.session_state.messages:
    with st.chat_message(message["role"]): #Die Rolle beschreibt von wem die Nachricht kommt und wird mit dem Streamlit widget direkt auch so angezeigt
        st.markdown(message["content"]) #markdown steht für die Formatierung des codes,

with col2:
    tempslider = 0.1*st.slider("Creativeness of AI", 0, 10, 7) #Hier ist der Slider welcher die Kretivität der KI bestimmt

with col1:
    if st.button("STOP!"):     #Hier ist der Stop-Knopf programmiert
        st.stop


def get_text_embedding(input_text: str):  #Das hier ist der Code, welcher von MistralAI vorprogrammiert ist, er ermöglicht mithilfe der MistralAI API eine Worteinbettung macht
    embeddings_batch_response = client.embeddings.create(  #Embeddings/Worteinbettungen ermöglichen NLPs besseres Verständnis/Kontextualisierung des Textes
          model = "mistral-embed",
          inputs = input_text
      )
    return embeddings_batch_response.data[0].embedding


def rag_pdf(pdfs: list, question: str) -> str:  #Hier findet der RAG Prozess des PDFs statt
    chunk_size = 4096
    chunks = []
    for pdf in pdfs:
        chunks += [pdf[i:i + chunk_size] for i in range(0, len(pdf), chunk_size)]

    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    question_embeddings = np.array([get_text_embedding(question)])
    D, I = index.search(question_embeddings, k = 4)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    text_retrieved = "\n\n".join(retrieved_chunk)
    return text_retrieved

def ask_mistral(messages: list, pdfs_bytes: list):  #Hier ist die Funktion welche die Anfrage/UserFrage an den MistralAI API geschickt
    if pdfs_bytes:  #PDF-Anfrage Teil
        pdfs = []
        for pdf in pdfs_bytes:
            reader = PyPDF2.PdfReader(pdf)
            txt = ""
            for page in reader.pages:
                txt += page.extract_text()
            pdfs.append(txt)
        messages[-1]["content"] = rag_pdf(pdfs, messages[-1]["content"]) + "\n\n" + messages[-1]["content"]
    resp = client.chat.stream(
    model="open-mistral-7b",
    messages=messages, #Textmessage Teil
    max_tokens=2048,  #EIn Maximallimit zu setzten ist immer schlau, falls ein Error passiert hat man so ein Sicherungsnetz, dass es nicht zu viele Tokens aufbraucht
    temperature=tempslider) #Hier wird die Temperaturwert an die Maschine geschickt
    for chunk in resp:          #Hier wird die Antwort ausgelesen
        yield chunk.data.choices[0].delta.content

if prompt := st.chat_input("Sprich zu Mistral!"):  #Hier ist die ChatBar mit welcher der User im UI inteagiert
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "safe_prompt": True})

    with st.chat_message("assistant"):
        response_generator = ask_mistral(st.session_state.messages, st.session_state.pdfs)
        response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": response})

uploaded_file = st.file_uploader("Choose a file", type=["pdf"])  #PDF upload Coe mithilfe von IO
if uploaded_file is not None:
    bytes_io = io.BytesIO(uploaded_file.getvalue())
    st.session_state.pdfs.append(bytes_io)



