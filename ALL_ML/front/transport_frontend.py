import streamlit as st
import requests

def check_transport():
    smart_api = 'http://127.0.0.1:8000/transport/check_transports'
    st.title('Transport')
    st.write('Загрузите изображение')
    upload_file = st.file_uploader('Выберите изображение')
    if upload_file is not None:
        st.image(upload_file, caption='Загруженное изображение')
        if st.button('Определить вид транспорта'):
            try:
                file = {
                    'image': (upload_file.name, upload_file.getvalue(), upload_file.type)
                }
                response = requests.post(smart_api, files=file)
                if response.status_code == 200:
                    result = response.json()
                    predict = result.get('this class')
                    st.success(f'Модель определил что этот класс относится к: {predict}')

                else:
                    st.error(f'Ошибка сервера {response.status_code}')
                    st.write(response.text)
            except requests.exceptions.ConnectionError:
                st.error(f'Ошибка в FastAPI')
            except Exception as errors:
                st.error(f'Ошибка{errors}')



