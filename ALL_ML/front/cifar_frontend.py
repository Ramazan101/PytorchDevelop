import streamlit as st
import requests

def science_image():
    url = 'http://127.0.0.1:8000/cifar/'

    st.title('Cifar-10 classifier')
    st.write('Загрузите изображение')

    uploaded_file = st.file_uploader('Выберите изображение для распознавание')
    if uploaded_file is not None:
        st.image(uploaded_file, caption=
                 'Загруженное изображение')
        if st.button('Распознать что на изображении'):
            try:
                files = {
                    'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                r = requests.post('http://127.0.0.1:8000/cifar', files=files, timeout=10)

                if r.status_code == 200:
                    result = r.json()
                    prediction = result.get('image_name')
                    st.success(f'Модель думает, что это: {prediction}')
                else:
                    st.error(f'Ошибка сервера: {r.status_code}')
                    st.write(r.text)
            except requests.exceptions.ConnectionError:
                st.error('FastAPI сервер не запущен')
            except Exception as e:
                st.error(f'Ошибка: {e}')

