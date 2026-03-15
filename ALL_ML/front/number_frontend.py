import streamlit as st
import requests

def check_number():
    api_url = 'http://127.0.0.1:8000/predict_image/'
    st.title('MNIST Fashion Classifier')
    st.write('Загрузите изображение с одеждами')
    upload_file = st.file_uploader('Выберите изображение')

    if upload_file is not None:
        st.image(upload_file, caption='Загруженное изображение')
        if st.button('Определить одежду'):
            try:
                files = {
                    'image': (upload_file.name, upload_file.getvalue(), upload_file.type)                }
                response = requests.post(api_url, files=files, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get('class')
                    st.success(f'Модель думает что это: {prediction}')
                else:
                    st.error(f'Ошибка сервера {response.status_code}')
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error('FastAPI сервер не запушен')
            except Exception as e:
                st.error(f'Ошибка({e}')
