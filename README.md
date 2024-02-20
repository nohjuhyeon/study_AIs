# study_AIs
## Macchine Learning
<details open>
<summary></summary>
 
### Clustering
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Clustering : Kmeans](docs/MLs/clusterings/01_clustering_simple.ipynb)|군집화 모델링하기|cluster.Kmeans()|
|2|[Clustering : basic](docs/MLs/clusterings/02_iris_KMeans.ipynb)|군집화 모델 평가하기|inertia_|
</details>

### Regression
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Regression : Basic](docs/MLs/Regression/01_linearRegression_simple.ipynb)|회귀 모델링하기|Linear_model. LinearRegression()|
|2|[Regression : r2_score](docs/MLs/Regression/02_BreastCancerWisconsin_LinearRegression.ipynb)|회귀 모델 r2_score로 평가하기|r2_score()|
|2|[Regression : pickle](docs/MLs/Regression/03_BreastCancerWisconsin_LinearRegression_reuse_pickle.ipynb)|pickle 이용하여 회귀 모델 다시 사용하기|pickle.dump()<br>pickle.load()|
|4|[Regression : RandomForestRegressor](docs/MLs/Regression/04_BreastCancerWisconsin_RandomForestRegressor.ipynb)|RandomForestRegressor 사용하여 회귀 모델링하기|RandomForestRegressor()|
|5|[Regression : MSE](docs/MLs/Regression/05_BreastCancerWisconsin_LinearRegression_evaluation.ipynb)|mean_squared_error 사용하여 모델의 에러율 확인하기|mean_squared_error()|
</details>

### Classifications
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Classifications : Basic](docs/MLs/classifications/01_LogisticRegression_simple.ipynb)|분류 모델링하기|LogisticRegression|
|2|[Classifications : accurancy score](docs/MLs/classifications/02_Classification_TitanicFromDisaster_train_LogisticRegression.ipynb)|분류 모델 평가하기|accuracy_score|
|3|[Classifications : plot tree](docs/MLs/classifications/03_Classification_TitanicFromDisaster_train_DecisionTreeClassifier.ipynb)|분류 모델 해석하기|plot_tree|
</details>

### Feature Engineering
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Feature Engineering : Scaling, Encoding](docs/MLs/Feature_Engineering/01_TitanicFromDisaster_train_LogisticRegression_featureengin.ipynb)|수치형 데이터에 scaling 적용하기<br>범주형 데이터에 Encoding 적용하기|MinMaxScaler, OneHotEncoder|
|2|[Feature Engineering : Cross Validation](docs/MLs/Feature_Engineering/02_SpineSurgeryList_GridSearchCV.ipynb)|학습데이터 교차검증하기|GridSearchCV|
|3|[Feature Engineering : Sampling](docs/MLs/Feature_Engineering/03_iris_samplings.ipynb)|범주형 데이터 resampling하기|over sampling : SMOTE<br>under sampling : NearMiss<br>Combine sampling : SMOTEENN|
</details>

</details>

## Natural Language Processing
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[NLP : Word Cloud](docs/NLPs/01_wordcloud_simple.ipynb)|Word Cloud 만들기|WordCloud()|
|2|[NLP : Word Cloud by Korean](docs/NLPs/02_wordcloud_korean.ipynb)|한글로 된 Word Cloud 만들기|
|3|[NLP : regex](docs/NLPs/03_wordcloud_korean_regexp.ipynb)|regex를 사용하여 조사 없애기|re.sub()|
|4|[NLP : soynlp](docs/NLPs/04_wordcloud_korean_soynlp.ipynb)|soynlp를 사용하여 명사만 추출하기|
|5|[NLP : morphemes](docs/NLPs/05_morphemes.ipynb)|Mecab, Okt로 자연어의 형태 분류하기|konlpy.tag : Okt<br> mecab : MeCab|
|6|[NLP : Mecab](docs/NLPs/06_wordcloud_korean_mecab.ipynb)|자연어에서 Mecab으로 명사만 추출하기|mecab.nouns()|
|7|[NLP : tokenizer](docs/NLPs/07_wordcloud_korean_tokenizers.ipynb)|문서 벡터화하기|TfidfVectorizer()|
|8|[NLP : 자연어 처리](docs/NLPs/08_NLP_classification_ynanewstitles.ipynb)|ynanews 자연어 처리하기|
</details>

## Numpy
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Numpy : Array](docs/Numpys/numpys.py)|Numpy의 배열 기능 사용하기|np.array|
</details>

 ## Quest
<details open>
<summary></summary>
 
 ### Machine Learning
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Regression](docs/quests/MLs/RentalOfContractType.ipynb)|Linear Regression 사용하여 결측치 채우기|
|2|[Feature Engineering : Scaling, Encoding](docs/quests/MLs/SpineSurgeryList_FeatureEngine.ipynb)|Scaling, Encoding을 통해 모델링 성능 향상시키기|
|3|[Feature Engineering : Resampling](docs/quests/MLs/SpineSurgeryList_GridSearchCV_resampling.ipynb)|Resampling을 통해 모델링 성능 향상시키기|
</details>

 ### Natural Language Processing
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Word Cloud](docs/quests/NLPs/wordcloud_regexp.ipynb)|자연어 처리하여 Word clound 만들기|
|2|[Classification : News](docs/quests/NLPs/classification_news.ipynb)|자연어 처리하여 뉴스 토픽 별 분류하기|
|3|[Classification : Reviews](docs/quests/NLPs/classification_movies.ipynb)|자연어 처리하여 리뷰 긍정,부정 분류하기|
|4|[Similiarity : complaints](docs/quests/NLPs/TDM_similiarity_seoul120.ipynb)|유사한 민원 찾기|

</details>

### Numpy
<details open>
<summary></summary>

|구분|이름|설명|비고|
|--|--|--|--|
|1|[Numpy : Array](docs/quests/Numpys/numpys.py)|Numpy의 Array 기능 활용하기|
</details>

</details>
