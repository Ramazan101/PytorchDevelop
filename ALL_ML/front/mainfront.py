import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if ROOT not in sys.path:
    sys.path.insert(0,str(ROOT))

import streamlit as st
from ALL_ML.front.frontend import check_image
from ALL_ML.front.number_frontend import check_number
from ALL_ML.front.cifar_frontend import science_image
from ALL_ML.front.flowers_frontend import flowers_images

with st.sidebar:
    name = st.radio(label='Models : ', options=['Info', 'Checking numbers', 'Guessing clothes', 'Cifar-10 classifier',
                                                'Flowers'])
if name == 'Info':
    st.title('Добро пожаловать')
    st.write('Checking numbers - Проверка цифр')
    st.write('Guessing clothes - Угадывание одежды')
    st.write('Cifar-10 classifier')
    st.write('Flowers')

elif name == 'Checking numbers':
    check_image()
elif name == 'Guessing clothes':
    check_number()
elif name == 'Cifar-10 classifier':
    science_image()
elif name == 'Flowers':
    flowers_images()