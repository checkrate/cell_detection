# cell_detection

1. Перед запуском программы следует разархивировать архив input, содержащий изображения, или создать папку input и, поместив ее в ту же директорию, что и программу, загрузить в нее 5 изображений.

![image](https://github.com/user-attachments/assets/e2edf458-6caa-44ec-abfe-0c43e3de3bfc)

2. После запуска и отработки программы, автоматически создается папка result, содержащая в себе каталоги со всеми результатами работы программы.

![image](https://github.com/user-attachments/assets/072b0c4a-aa82-4272-a309-0a325a425d74)

3. Каждый каталог измерения содержит в себе подкаталог, со всей информацией, касательно хода выполнения программы:

![image](https://github.com/user-attachments/assets/32f91fb4-3ed0-45fa-9524-abb2811f0724)

ROI - сетка, распознанная на изображении:

![17](https://github.com/user-attachments/assets/9f4fb586-3887-416e-a3b8-7bc77e835e39)

ROI_cropped - итоговое изображение, подаваемое на вход модели (обрезанное по границам сетки):

![17](https://github.com/user-attachments/assets/80f775f2-0c08-4f36-8c0a-053c332d4061)

detected - распознанные на изображении биологические клетки:

![17](https://github.com/user-attachments/assets/96da58e0-d1c4-45a9-866f-2c1e74fb99b3)

Файл result.txt содержит в себе данные об обнаруженных клетках на каждом изображении, и итоговый расчет:

![image](https://github.com/user-attachments/assets/1928631b-bd60-4a9c-b14d-50ef07d08aa4)




