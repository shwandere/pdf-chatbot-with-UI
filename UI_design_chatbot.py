#UI_design_chatbot
import urllib.request
import streamlit as st 
import urllib
import base64
from chatbot import pdf_chatbot

#function to display the PDF of a given file 
def displayPDF(file):

    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="950" type="application/pdf"></iframe>'

    pdf_box = st.sidebar
    pdf_box.markdown(pdf_display, unsafe_allow_html=True)

st.title("Hi this is pdf reader, upload pdf and ask question")
uploadbtn = st.button("Upload File")

#print(st.session_state)

uploaded_file = st.file_uploader("Choose a pdf",type='pdf')
print(uploaded_file)
if uploaded_file is not None:
    # print(uploaded_file)
    st.write("You selected the file:", uploaded_file.name)
    displayPDF(uploaded_file)

    #ques_ans = st.sidebar
    question_asked = st.text_input("Ask your question")
    check_submit = st.button("Submit")

    if check_submit is not None:
        answer_to_return = st.text_input("Here is your answer")
        answer_to_return = pdf_chatbot(question_asked,uploaded_file)


