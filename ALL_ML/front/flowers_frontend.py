import streamlit as st
import requests

def flowers_images():
    api = 'http://127.0.0.1:8000/flowers/check_flowers'

    st.title('Flowers Image')
    st.write('Загрузите изображение')

    upload_file = st.file_uploader('Выберите изображение')

    if upload_file is not None:
        st.image(upload_file, caption='Загруженное изображение')

        if st.button('Определить цветок'):
            try:
                files = {
                    'image': (upload_file.name, upload_file.getvalue(), upload_file.type)
                }

                response = requests.post(api, files=files, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    pred = result.get('this class')
                    st.success(f'Модель думает, что это относится к: {pred}')
                else:
                    st.error(f'Ошибка сервера: {response.status_code}')
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error('FastAPI сервер не запущен')
            except Exception as e:
                st.error(f'Ошибка: {e}')