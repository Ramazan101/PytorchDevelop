import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if ROOT not in sys.path:
    sys.path.insert(0,str(ROOT))

import streamlit as st
from ALL_ML.front.frontend import check_image
from ALL_ML.front.number_frontend import check_number

with st.sidebar:
    name = st.radio(label='Models : ', options=['Info', 'Checking numbers', 'Guessing clothes'])
if name == 'Info':
    st.title('Добро пожаловать')
    st.write('Checking numbers - Проверка цифр')
    st.write('Guessing clothes - Угадывание одежды')

elif name == 'Checking numbers':
    check_image()
elif name == 'Guessing clothes':
    check_number()