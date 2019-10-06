# severstal-comp

В соревновании требуется предоставить решение сегментации дефектов на листовой стали (царапины, вмятины, потертости и точки). Основная проблема - это плохая разметка, которая осуществлялась разными людьми. Кто-то обводил объекты пиксель-в-пиксель, а кто-то многоугольниками, площадь которых на 80% незатронута (я сам удивлен, особенно в свете того, что северсталь потратит минимум 120к на соревнование). 

Характеристика самого дефекта - это отношение длины контура к его площади.

Я реализовал эту идею в лосс функции при помощи операторов Собеля. Добавка этой характеристики в лосс к кросс энтропии пинализирует модель, если а) она находит отдельно стоящие дефекты б) делает контур слишком сложным .

Оценить вклад добавки и подобрать ее вес в лосс функции можно сравнивая коэффициенты Дайса на кросс-валидации. 

