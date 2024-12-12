import os
from PIL import Image
from model_utils import MiniCPM_model

def main():
    cache_directory = "model_cache"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    model_name = "openbmb/MiniCPM-V-2_6"
    api_key = os.getenv("HF_API_TOKEN")
    model = MiniCPM_model(cache_directory, model_name, api_key)

    image_path = "example_docs/2-1.jpg"
    image = Image.open(image_path).convert("RGB")

    question = """
    Пожалуйста собери следующую информацию с документа: 
    Фамимлия
    Имя
    Отчество 
    Верни ответ в виде json файла с полями и ответами на них.
    """
    msgs = [{"role": "user", "content": [image, question]}]

    model_answer = model.chat(image=None, msgs=msgs)

    print("Ответ модели:")
    print(model_answer)

if __name__ == "__main__":
    main()