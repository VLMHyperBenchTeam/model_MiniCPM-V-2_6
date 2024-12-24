# MiniCPM-V_2.6

Mini-CPM (Mini Chinese Pretrained Model)

Материалы:
* [ссылка на GitHub](https://github.com/OpenBMB/MiniCPM-V?tab=readme-ov-file#minicpm-v-26) 
* [ссылка на HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-2_6)

## Docker контейнер модели

### Build Docker image

Для сборки `Docker image` выполним команду:
```
docker build -t ghcr.io/vlmhyperbenchteam/mini_cpm:latest -f docker/Dockerfile .
```

### Run Docker Container

Для запуска `Docker Container` выполним команду:
```
docker run \
    --gpus all \
    -it \
    -v ./src:/workspace \
    ghcr.io/vlmhyperbenchteam/mini_cpm:latest
```

Нам откроется терминал внутри `Docker Container`.

Для запуска предсказаний выполним в нем команду:
```
cd cd workspace
python run_predict.py
```

## Ключевые особенности модели MiniCPM-V 2.6
