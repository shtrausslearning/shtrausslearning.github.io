---
title: Recommendation Systems Course Work
hide_comments: false
---

# **Recsys**

## <b><span style='color:#3489c2'>От классического ML к персонализации</span></b> 


<div class="grid cards" markdown>

- <center><b>[Сегментация Пользователей (Эвристическая, Кластеризация, Лог Регрессия)](User_Segmentation)</b></center>
- <center><b>[Look-a-like Сегментация Пользователей](Look-a-like_Segmentation)</b></center>
- <center><b>[Next Best Action Моделирование](NBA_Modeling)</b></center>
- <center><b>[Uplift Моделирования](Uplift_Modeling)

	--- 
	
	Посмотрим на метод который позволяет оценить эффект от воздействия определенной действия на целевую аудиторию. Он помогает определить, какие клиенты действительно выиграют от рекомендации, а не просто отреагируют на него, что позволяет более эффективно распределять ресурсы и повышать рентабельность инвестиций

</div>


## <b><span style='color:#3489c2'>Классические методы рекомендаций</span></b> 

`Эвристические`, `apriori` и подходы `матричной факторизации`

<div class="grid cards" markdown>

- <center><b>[Эвристические Модели,Коллаборативная Фильтрация](recsys_rule_knn)</b></center>
- <center><b>[Модели Матричной Факторизации](recsys_matrix_decomposition)</b></center>
- <center><b>[Модели Матричной Факторизации Практика](recsys_matrix_decomposition_practice)</b></center>
- <center><b>[A/B - тесты в рекомендательных системах](AB_seminar)</b></center>

</div>


## <b><span style='color:#3489c2'>Контентные и гибридные методы рекомендаций</span></b> 


Подходы на основе `LightFM` и много этапные подходы для решения задач рекомендации  

<div class="grid cards" markdown>

- <center><b>[Контентные методы рекомендаций](recsys_lightfm)</b></center>
- <center><b>[Рекомендации по текстовому описанию](recsys_nlp)</b></center>
- <center><b>[Ранжирование каталога товаров I](Ranker_Regressor)</b></center>
- <center><b>[Ранжирование каталога товаров II](Ranker_CatBoost)</b></center>
- <center><b>[Практическое занятие по рекомендательным системам. Двухуровневая модель](2stagerecsys)</b></center>

</div>


## <b><span style='color:#3489c2'>Современные методы рекомендаций</span></b> 

Нейросетевые подходы и подобные

<div class="grid cards" markdown>

- <center><b>[Нейросетевые методы рекомендаций (DSSM)](dssm-towers)</b></center>
- <center><b>[Многорукие бандиты для оптимизации A/B - тестирования]()</b></center>
- <center><b>[Многорукие бандиты для задачи рекомендации]()</b></center>

</div>


## <b><span style='color:#3489c2'>Проекты</span></b> 
 

<div class="grid cards" markdown>

- <center><b>[X-Learner Uplift](prob_x5)</b></center>

	---
	
	Цель: реализовать самописный код для модели X-learner. 

- <center><b>[Модели матричной факторизации](prob_mf)</b></center>

	Цель: Обучение рекомендательный моделей и сравнения их качество на основе три критерии; классификации (`hitrate`), ранжирования (`MRR`,`NDCG`) и разнообразия (`coverage`)

- <center><b>[Контентные методы рекомендаций](prob_lightfm)</b></center>

- <center><b>[Гибридные подход рекомендаций](prob_hybrid)</b></center>

- <center><b>[Нейросетевые методы рекомендаций (NeuMF)](prob_neumf)</b></center>

</div>