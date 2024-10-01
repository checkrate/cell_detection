#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
import torch 
import os, sys
from math import *
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import datetime


# In[93]:


FOLDER_PATH = 'D:/celldetect/input'
RESULT_PATH = 'D:/celldetect/results'
YOLO_PATH = 'D:/celldetect/best.pt'


def preprocessing(image):
    '''
    Эта функция выполняет предварительную обработку изображения:
    - Преобразует изображение в градации серого
    - Применяет размытие по Гауссу
    - Выполняет адаптивное пороговое преобразование
    - Находит контуры и рисует их на исходном изображении

    Аргументы:
    - image: входное изображение в формате BGR

    Возвращает:
    - image: изображение с нарисованными контурами
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5:
            cv2.drawContours(image, contours, c, (0, 255, 0), 2)
        c += 1
    return image


# Функция обнаружения линий
def lines_detect(image):
    '''
    Эта функция обнаруживает линии на изображении с помощью преобразования Хафа.

    Аргументы:
    - image: входное изображение в формате BGR

    Возвращает:
    - lins: список обнаруженных линий в формате namedtuple Line (start, end)
    '''
    from collections import namedtuple
    Line = namedtuple('Line', ['start', 'end'])
    lins = []
    edges = cv2.Canny(image, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=450)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        start_coordinates = (x1, y1)
        end_coordinates = (x2, y2)
        new_line = Line(start_coordinates, end_coordinates)
        lins.append(new_line)
    return lins


# Функция сортировки линий
def sort_lines(lines):
    '''
    Эта функция сортирует линии на горизонтальные и вертикальные.

    Аргументы:
    - lines: список линий в формате namedtuple Line (start, end)

    Возвращает:
    - horizontal_lines: отсортированный список горизонтальных линий
    - vertical_lines: отсортированный список вертикальных линий
    '''
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1 = line.start
        x2, y2 = line.end

        angle = atan2(y2 - y1, x2 - x1)

        if abs(angle) < pi / 4 or abs(angle) > 3 * pi / 4:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)
        horizontal_lines = sorted(horizontal_lines, key=lambda line: line.start[1])
        vertical_lines = sorted(vertical_lines, key=lambda line: line.start[0])

    return horizontal_lines, vertical_lines


# Функции для вычисления расстояний между линиями
def x_distances(line1, line2):
    '''
    Эта функция вычисляет абсолютное расстояние по оси X между двумя линиями.

    Аргументы:
    - line1: первая линия в формате namedtuple Line (start, end)
    - line2: вторая линия в формате namedtuple Line (start, end)

    Возвращает:
    - расстояние по оси X между начальными точками двух линий
    '''
    return abs(line1.start[0] - line2.start[0])

def y_distances(line1, line2):
    '''
    Эта функция вычисляет абсолютное расстояние по оси Y между двумя линиями.

    Аргументы:
    - line1: первая линия в формате namedtuple Line (start, end)
    - line2: вторая линия в формате namedtuple Line (start, end)

    Возвращает:
    - расстояние по оси Y между начальными точками двух линий
    '''
    return abs(line1.start[1] - line2.start[1])


def remove_small_keys(d):
    '''
    Эта функция удаляет из словаря ключи, которые содержат менее трех элементов.

    Аргументы:
    - d: словарь, из которого необходимо удалить маленькие ключи

    Возвращает:
    - d: словарь, содержащий только ключи с тремя или более элементами
    '''
    keys = [k for k in d.keys()]
    max_key = max(keys)
    if len(d[max_key]) >= 3:
        return {max_key: d[max_key]}
    else:
        return d


# In[99]:


def preprocessing_yolo(image):
    '''
    Эта функция выполняет предварительную обработку изображения для YOLO:
    - Изменяет размер изображения до 640x640
    - Нормализует изображение
    - Преобразует изображение в тензор PyTorch


    Аргументы:
    - image: входное изображение в формате BGR

    Возвращает:
    - input_tensor: тензор, пригодный для входа в модель YOLO
    - resized_image: изображение после изменения размера
    '''
    resized_image = cv2.resize(image, (640, 640))
    normalized_image = resized_image / 255.0
    input_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0).float()
    return input_tensor, resized_image


# In[107]:


# Получение списка файлов в папке
file_list = os.listdir(FOLDER_PATH)

# Инициализация модели YOLO
model = YOLO(YOLO_PATH)

# Инициализация пустого списка для хранения количества обнаруженных клеток
cells = []

# Получение текущей даты и формирование уникального имени директории для сохранения результатов
current_date = datetime.datetime.now()
date_str = current_date.strftime('%Y_%m_%d')
num_runs_today = len([name for name in os.listdir(RESULT_PATH) if date_str in name]) + 1
new_dir_name = f'{date_str}_{num_runs_today}'
new_dir_path = os.path.join(RESULT_PATH, new_dir_name)

# Создание директории для сохранения результатов
os.makedirs(new_dir_path, exist_ok=True)

# Создание поддиректорий для различных типов выходных данных
roi_dir = os.path.join(new_dir_path, 'ROI')
os.makedirs(roi_dir, exist_ok=True)

roi_cropped_dir = os.path.join(new_dir_path, 'ROI_cropped')
os.makedirs(roi_cropped_dir, exist_ok=True)

detected_dir = os.path.join(new_dir_path, 'detected')
os.makedirs(detected_dir, exist_ok=True)

# Инициализация списка для хранения количества клеток на каждом изображении
cell_counts = []

# Цикл обработки каждого файла из списка
i = 0
for file_name in file_list:
    i += 1
    j = 0

    # Загрузка изображения и его копирование для обработки
    image = cv2.imread(os.path.join(FOLDER_PATH, file_name))
    image0 = image.copy()

    # Предварительная обработка изображения (поиск и отрисовка контуров)
    image = preprocessing(image)

    # Сортировка найденных линий на горизонтальные и вертикальные
    horizontals, verticals = sort_lines(lines_detect(image))

    # Создание словарей для хранения расстояний между горизонтальными и вертикальными линиями
    distances_horizontal_dict, distances_vertical_dict = {}, {}

    # Вычисление расстояний между горизонтальными линиями
    for i in range(len(horizontals) - 1):
        distance = y_distances(horizontals[i], horizontals[i + 1])
        if distance not in distances_horizontal_dict:
            distances_horizontal_dict[distance] = [horizontals[i], horizontals[i + 1]]
        else:
            distances_horizontal_dict[distance].extend([horizontals[i], horizontals[i + 1]])

    # Вычисление расстояний между вертикальными линиями
    for i in range(len(verticals) - 1):
        distance = x_distances(verticals[i], verticals[i + 1])
        if distance not in distances_vertical_dict:
            distances_vertical_dict[distance] = [verticals[i], verticals[i + 1]]
        else:
            distances_vertical_dict[distance].extend([verticals[i], verticals[i + 1]])

    # Удаление ключей с маленьким количеством элементов
    distances_horizontal_dict = remove_small_keys(distances_horizontal_dict)
    distances_vertical_dict = remove_small_keys(distances_vertical_dict)

    # Получение максимальных двух расстояний для горизонтальных и вертикальных линий
    max_horizontal = sorted(distances_horizontal_dict.keys())[::-1][:2]
    max_vertical = sorted(distances_vertical_dict.keys())[::-1][:2]

    # Извлечение линий, соответствующих максимальным расстояниям
    lines_vert = [value for key, value in distances_vertical_dict.items() if key in max_vertical]
    lines_vert = [item for sublist in lines_vert for item in sublist]
    lines_horiz = [value for key, value in distances_horizontal_dict.items() if key in max_horizontal]
    lines_horiz = [item for sublist in lines_horiz for item in sublist]

    # Сортировка линий и выбор крайних для определения области ROI
    lines_vert.sort(key=lambda line: line.start[0])
    lines_vert = [lines_vert[0], lines_vert[-1]]
    lines_horiz.sort(key=lambda line: line.start[1])
    lines_horiz = [lines_horiz[0], lines_horiz[-1]]

    # Определение области интереса (ROI)
    roi = image0[lines_horiz[0].start[1]:lines_horiz[1].start[1], lines_vert[0].start[0]:lines_vert[1].end[0]].copy()
    save_path = os.path.join(roi_cropped_dir, f'{i}.jpg')
    cv2.imwrite(save_path, roi)
    
    # Рисование вертикальных и горизонтальных линий на исходном изображении
    for line in lines_vert:
        cv2.line(image0, line.start, line.end, (0, 0, 255), 2)
    for line in lines_horiz:
        cv2.line(image0, line.start, line.end, (0, 0, 255), 2)

    # Сохранение изображения с отмеченными линиями в папке ROI
    save_path = os.path.join(roi_dir, f'{i}.jpg')
    cv2.imwrite(save_path, image0)

    # Предварительная обработка области ROI для модели YOLO
    input_tensor, resized_roi = preprocessing_yolo(roi)

    # Обнаружение объектов с использованием модели YOLO
    with torch.no_grad():
        outputs = model(input_tensor)

    # Получение предсказанных объектов
    predictions = outputs[0].boxes
    boxes = predictions.xyxy
    scores = predictions.conf
    class_ids = predictions.cls.long()

    # Настройки порога уверенности и цвета рамки
    confidence_threshold = 0.5
    color = (0, 255, 0)
    thickness = 2

    # Отрисовка рамок вокруг обнаруженных клеток, соответствующих классу "клетка"
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > confidence_threshold and class_id == 0:
            x1, y1, x2, y2 = box.numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(resized_roi, (x1, y1), (x2, y2), color, thickness)
            j += 1

    # Сохранение изображения с обнаруженными клетками в папке detected
    cell_counts.append(j)
    save_path = os.path.join(detected_dir, f'{i}.jpg')
    cv2.imwrite(save_path, resized_roi)

# Запись результатов в текстовый файл
output_file = os.path.join(new_dir_path, 'result.txt')
cells = 0
with open(output_file, 'w') as f:
    for i, count in enumerate(cell_counts, start=1):
        f.write(f'Количество клеток на {i} изображении: {count}\n')
        print(count)
        cells += count
    f.write(f'Всего распознано клеток: {cells}\n')
    f.write(f'Итоговое количество клеток: {(cells * 25000) / 20}\n')

