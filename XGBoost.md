## XGBoost Info

### BEFORE SETTINGS, 파라미터 튜닝도 중요하지만 피처 엔지니어링에 더 집중하는 것이 좋은 결과를 가져온다. 

### Control Overfitting (과대적합 조정)

> 1. 모델 복잡도 개선 -> max_depth, min_child_weight_gamma 조정 
>
> 2. noise 를 균일하게 만들기 위해서 randomness 추가 -> subsample colsample_bytree 
>
>    eta 줄이기 + num_round는 증가시킬 것 

### Handle Imbalanced Dataset 

> 1. AUC의 성능을 올리고 싶다면 -> scale_via_weight 조정 
> 2. right probability(정규분포?)  -> max_delta_step 지정 

### set the XGBoost parameter 

> [python]
>
> import xgboost as xgb
>
> ----------------------------------
>
> ##### read in data
>
> dtrain = xgb.DMatrix(...)
>
> dtest = xgb.DMatrix(...)
>
> ---------------------
>
> ##### specify parameters via map
>
> param = {'max_depth': 4, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic' }
>
> num_round = 100
>
> bst = xgb.train(param, dtrain, num_round)
>
> ---------------------------------------------------
>
> ##### make prediction
>
> preds = bst.predict(dtest)



□ General Parameters : XGBoost의 전반적인 기능을 정의함.

 > booster [default=gbtree] >> 일반적으로 gbtree의 성능이 낫다.

   - gbtree: tree-based models

   - gblinear: linear models

 > silent [default=0]

   - 1: 동작 메시지를 프린트하지 않음. 



□ Booster Parameters (아래는 gbtree booster 기준으로 정리되어있음.)

 > eta [default=0.3] => learning_rate

   - 딥러닝의 learning rate와 같은 개념. 값이 너무 높으면 학습이 잘 안되고, 
   
   - 값이 너무 낮으면 학습이 느릴 수 있다.

   - 각 단계에서 가중치를 줄임으로써 모델을 더 강건하게 만든다.

   - 일반적으로 0.01-0.2

 > min_child_weight [default=1] (Should be tuned using CV)

   - child의 관측(?)에서 요구되는 최소 가중치의 합

   - over-fitting vs under-fitting을 조정하기 위한 파라미터.

   - 너무 큰 값이 주어지면 under-fitting.

 > max_depth [default=6] (Should be tuned using CV)

   - 트리의 최대 깊이. 값이 높을 수록 더 복잡한 트리모델을 생성하고, 지나칠 경우 과적합의 원인이 된다. 

   - 일반적으로 3-10

 > max_leaf_nodes

   - 최종 노드의 최대 개수. (max number of terminal nodes)

   - 이진 트리가 생성되기 때문에 max_depth가 6이면 max_leaf_nodes는 2^6개가 됨.

 > gamma [default=0]

   - 분할을 수행하는데 필요한 최소 손실 감소를 지정한다.

   - 알고리즘을 보수적으로 만든다. loss function에 따라 조정해야 한다.

 > subsample [default=1]

   - 각 트리마다의 관측 데이터 샘플링 비율.

   - 값을 적게 주면 over-fitting을 방지하지만 값을 너무 작게 주면 under-fitting.

   - 일반적으로 0.5-1

 > colsample_bytree [default=1]

   - 트리를 생성할 때 훈련 데이터에서 변수를 샘플링해주는 비율.
   
   - 모든 트리는 전체 변수의 일부만을 학습하여 서로의 약점을 보완해준다. 

   - 일반적으로 0.5-0.9
   
 > colsample_bylevel 
 
   - 트리의 레벨 별로 훈련 데이터의 변수를 샘플링해주는 비율.
   
   - 일반적으로 0.6 - 0.9 값 사용. 

 > lambda [default=1] => reg_lambda

   - 가중치에 대한 L2 정규화 용어 (Ridge 회귀 분석과 유사(?))

 > alpha [default=0] => reg_alpha

   - 가중치에 대한 L1 정규화 용어 (Lasso 회귀 분석과 유사(?))

 > scale_pos_weight [default=1]

   - 불균형한 경우 더 빠른 수렴(convergence)에 도움되므로 0보다 큰 값을 쓸것.
   
 



□ Learning Task Parameters : 각 단계에서 계산할 최적화 목표를 정의하는 데 사용된다.

 > objective [default=reg:linear]

   - binary:logistic : 이진 분류를 위한 로지스틱 회귀, 예측된 확률을 반환한다. (not class)

   - multi:softmax : softmax를 사용한 다중 클래스 분류, 예측된 클래스를 반환한다. (not probabilities)

   - multi:softprob : softmax와 같지만 각 클래스에 대한 예상 확률을 반환한다.

 > eval_metric [default according to objective]

   - 회귀 분석인 경우 'rmse'를, 클래스 분류 문제인 경우 'error'를 default로 사용.

   - rmse : root mean square error

   - mae : mean absolute error

   - logloss : negative log-likelihood

   - error : Binary classification error rate (0.5 threshold)

   - merror : Multiclass classification error rate

   - mlogloss : Multiclass logloss

   - auc : Area under the curve

 > seed [default = 0]

   - 난수 시드

   - 재현 가능한 결과를 생성하고 파라미터 튜닝에도 사용할 수 있다.

     

> > 출처 : http://okminseok.blogspot.com/2017/09/ml-xgboost.html
> >
> > 참고 영상: https://www.youtube.com/watch?v=Dhwmd_IyW3g(14:02~)
