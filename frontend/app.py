import requests
import streamlit as st

# Укажи здесь IP своей ВМ в Yandex Cloud
VM_PUBLIC_IP = "130.193.57.190" 
BASE_URL = f"http://{VM_PUBLIC_IP}:8000"

st.title("YOLO & RAG")

tab1, tab2 = st.tabs(['📷 Image Classification', '💬 RAG'])

def main():
    with tab1:
        st.header("YOLOv11 Object Detection")
        # Поле загрузки изображения
        image_file = st.file_uploader("Upload an image for YOLO", type=['jpg', 'jpeg', 'png'])
        
        if st.button("Detect Objects") and image_file is not None:
            # Отображаем картинку
            st.image(image_file, caption="Uploaded Image", use_container_width=True)
            
            # Подготовка файла для отправки
            files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
            
            try:
                # Отправка запроса на FastAPI
                res = requests.post(f"{BASE_URL}/clf_image", files=files)
                
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Detected: **{data['class_name']}** (Index: {data['class_index']})")
                else:
                    st.error(f"Error: {res.status_code}. Objects might not be found.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    with tab2:
        st.header("Real Estate Expert Chat")
        st.write("Ask about prices, areas, or specific apartment details.")
        
        # Поле ввода вопроса
        user_query = st.text_input('Your question to AI Expert:', placeholder="Find the cheapest apartment...")
        
        if st.button('Ask Expert'):
            if user_query:
                # Формируем JSON согласно схеме TextInput
                payload = {'question': user_query}
                
                with st.spinner("Expert is analyzing market data..."):
                    try:
                        res = requests.post(f"{BASE_URL}/rag", json=payload)
                        
                        if res.status_code == 200:
                            answer = res.json()
                            st.markdown("### Expert Analysis:")
                            st.write(answer['text'])
                        else:
                            st.error(f"Server returned an error: {res.status_code}")
                    except Exception as e:
                        st.error(f"Could not connect to RAG service: {e}")
            else:
                st.warning("Please enter a question first.")

if __name__ == '__main__':
    main()