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
from ALL_ML.front.smartphones_frontend import check_smartphones
from ALL_ML.front.transport_frontend import check_transport
from ALL_ML.front.cifar100_frontend import check_cifar100

with st.sidebar:
    name = st.radio(label='Models : ', options=['Info', 'Checking numbers', 'Guessing clothes', 'Cifar-10 classifier',
                                                'Flowers', 'Smartphones', 'Transport', 'Cifar100'])
if name == 'Info':
    st.title('Добро пожаловать')
    st.write('Checking numbers - Проверка цифр')
    st.write('Guessing clothes - Угадывание одежды')
    st.write('Cifar-10 classifier')
    st.write('Flowers')
    st.write('Smartphones')
    st.write('Transport')
    st.write('Cifar100')

elif name == 'Checking numbers':
    check_image()

elif name == 'Guessing clothes':
    check_number()

elif name == 'Cifar-10 classifier':
    science_image()

elif name == 'Flowers':
    flowers_images()

elif name == 'Smartphones':
    check_smartphones()

elif name == 'Transport':
    check_transport()

elif name == 'Cifar100':
    check_cifar100()
