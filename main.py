import streamlit as st
from data_preparation import prepare_data
from workflow import app as workflow_app

# Prepare the data
prepare_data()

# Streamlit app
st.title("RetailX AI Assistant")
st.write("Ask a question about RetailX customers, products, and sales:")

question = st.text_input("Question")
if st.button("Submit"):
    if question:
        inputs = {"question": question}
        result = workflow_app.invoke(inputs)
        st.write("Answer:", result['answer'])
    else:
        st.write("Please enter a question.")