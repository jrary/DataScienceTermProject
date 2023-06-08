# Data Science Termproject
#### 가천대학교 2023학년 1학기 데이터과학 
#### Theme Park visitor Prediction(서울대공원)

## Data References
1. [기상청 기상자료개방포털](https://data.kma.go.kr/data/rmt/rmtList.do?code=420&pgmNo=572)
2. [대기질측정정보](https://data.gg.go.kr/portal/data/service/selectServicePage.do?page=1&rows=10&sortColumn=&sortDirection=&infId=GE0DUHTX3VX0GL4R0LUS26448884&infSeq=1&order=)
3. [서울대공원 일일입장객수 정보](http://data.seoul.go.kr/dataList/OA-15386/F/1/datasetView.do)

## How to run?
Just run these files.
* ensemble_gradient_boost.ipynb
* regression_model.ipynb

## Architecture
![](https://github.com/wooing1084/DataScienceTermProject/blob/main/imgs/architecture_overall.png?raw=true)

## Used algorithms

### Regression
Generalized Additive Model (GAM)
### Ensemble Learning
Gradient boosting Regressor

## Result
* Optimal hyperparameter
  * n_estimators : 200
  * max_depth : 3
  * learning_rate : 0.1
* Feature importance
![](https://github.com/wooing1084/DataScienceTermProject/assets/32007781/8bf2be4b-54dd-44be-95ac-edef77eee974)
* Accuracy

![](https://github.com/wooing1084/DataScienceTermProject/blob/main/imgs/Ensemble_accuracy.png?raw=true)

