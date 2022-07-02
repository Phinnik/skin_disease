# Skin disease

Проект по классификации кожных заболеваний

[Подробнее о проекте](docs/model_card.md)

По вопросам обращаться: tg: [@phinnik](https://t.me/phinnik)

---
## Инструкции

### Установка библиотек
```shell
pip install -r requirements.txt
```
Поменяйте версию pytorch в зависимости от ваших драйверов.

### Данные для обучения
#### skin-cancer-mnist-ham10000

1. Cкачать архив с данными на [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Создать директорию `data/raw/skin_cancer_mnist` и распаковать в нее содержимое архива
3. Сгенерировать сплит с помощью скрипта `python src/scripts/generate_split.py`

### Запуск обучения
Для этого вам необходимо будет пройти [регистрацию в ClearMl](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/) (или развернуть его локально)

```shell
python models/clf/train.py
```

### Запуск сервиса
#### В контейнере

```shell
cd docker-compose
docker compose -f deploy.yml up --build
```
Напоминалка: [установить nvidia-docker](sudo systemctl restart docker)


#### Вручную
```shell
export RELEASE_VERSION=0.1.0
cd src/app
uvicorn main:app --reload --host 0.0.0.0
```


### Отправка запроса к сервису
#### С помощью консоли
```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@data/raw/skin_cancer_mnist/ham10000_images_part_2/ISIC_0029306.jpg;type=image/jpg'
```

#### python
```python
>>> import requests

>>> url = 'http://127.0.0.1:8000/classify'
>>> image_fp = 'data/raw/skin_cancer_mnist/ham10000_images_part_2/ISIC_0029306.jpg'
>>> with open(image_fp, 'rb') as f:
>>>     image = f.read()

>>> requests.post(url, files={'image': image}).json()
{'Pathology': 'Меланоцитарные невусы'}
```


### Метрик тест
```shell
python src/scripts/calculate_metrics.py --split_dir data/processed/splits/skin_cancer_mnist_split --model_checkpoint_fp models/model_releases/0.1.0/models.pkl
```
