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

## Business objective
When people go out with their family or friends, people often go to amusement parks. However, there are so many people that you may not be able to ride the rides you want to ride or see what you want to see. The word "눈치게임" has even been coined to avoid this situation. We identified these problems and planned an amusement park visitor prediction model to provide convenience to people.


## Architecture
![](https://github.com/wooing1084/DataScienceTermProject/blob/main/imgs/architecture_overall.png?raw=true)

## Used algorithms

### Regression
Generalized Additive Model (GAM)
### Ensemble Learning
Gradient boosting Regressor

## Result
<details>
 <summary> Optimal hyperparameter </summary>
 n_estimators : 200, max_depth : 3, learning_rate : 0.1
 </details>


<details>
 <summary> Feature importances </summary>
 <img class="fit-picture" src="https://github.com/wooing1084/DataScienceTermProject/assets/32007781/8bf2be4b-54dd-44be-95ac-edef77eee974">
 
 </details>
 
<details>
 <summary> Accuracy </summary>
 <img class="fit-picture" src="https://github.com/wooing1084/DataScienceTermProject/blob/main/imgs/Ensemble_accuracy.png?raw=true">
 
 </details>

## Conclusion
The prediction accuracy is about 70 percent, so we think we can roughly see how much it comes numerically. However, We think it is a little difficult to know the degree of congestion by figures alone. Therefore, it would be more helpful if we created a congestion level.
<details>
<summary>Improvements</summary>
There were some improvements to be made while carrying out this project. Most of these were data set issues, including.
1.	In the past, many people visited Seoul Grand Park, but the number of visitors decreased over time. This phenomenon has led to a data imbalance.
  

2.	It does not properly reflect the expected days of many visitors, such as holidays and Children's Day. Although the day of year value was added to reflect this, it was not accurately reflected.
 

3.	No visitor data has been collected after the pendemic. The official Endemic Declaration has only been issued relatively recently. Therefore, the most recent trend was not reflected.


4.	Information about the park or surrounding events was not reflected. In the case of large parks, the number of visitors varies greatly depending on the event, but data could not be collected.

</details>
