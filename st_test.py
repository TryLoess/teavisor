import requests
import streamlit as st

if __name__ == "__main__":
    st.info("测试是否能连接上外网,访问url为http://43.136.22.64:5050/paddle_test_data，方式为post")
    response = requests.post("http://43.136.22.64:5050/paddle_test_data")
    st.info(response.status_code)