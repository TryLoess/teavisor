import requests
import streamlit as st

if __name__ == "__main__":
    response = requests.post("http://43.136.22.64:5050/paddle_test_data")
    st.info(f"回复的code为{response.status_code}")