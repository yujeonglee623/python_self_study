# 2025.11.28
## Decision Tree
### 노드 나누는 기준
- 결정트리는 지니 불순도(Gini)나 엔트로피(Entropy)가 더 감소하는지 보면서 정보가 더 좋은지 봄
- 지니 불순도가 높은 경우
    - 클래스가 섞여 있음  
        -> 더 나눠서 순수하게 만들려고 시도  
        -> 계속 분기(split)함
- 지니 불순도가 이미 낮은 경우
    - 거의 한 클래스만 존재  
        -> 나눠도 크게 개선되지 않음  
        -> 이 노드는 더 이상 나누지 않음(leaf)
- 추가로 나누지 않는 다른 이유들
    - 더 나눠도 지니 감소량이 거의 없을 때(정보 이득이 너무 작음) 
    - 노드 안 데이터수가 너무 적을 때  
        min_samplse_split, min_samples_leaf
    - 트리의 최대 깊이를 이미 다 썼을 때  
        max_depth
    - 모든 데이터가 같은 클래스일 때  
        value가 [0, 52] 같은 형태
## 앙상블 학습 모델 (Ensemble Learning Model)
### 앙상블 유형
- 보팅 (Voting)
- 배깅 (Bagging)
- 부스팅 (Boosting)
- 스태킹 (Stacking)  
=> 넓은 의미로 서로 다른 모델을 결합한 것들을 앙상블로 지칭
## 랜덤 포레스트 (Random Forest) - Bagging
- 다재다능한 알고리즘
- 앙상블 알고리즘 중 비교적 빠른 수행 속도를 가짐
- 다양한 영역에서 높은 예측 성능을 보임
- 여러개의 결정 트리 분류기가 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해서 개별적으로 학습 후 보팅을 통해 예측
##### 장점
- 많은 하이퍼 파라미터 튜닝을 거치지 않아도 일반적으로 안정적으로 좋은 성능 발휘
- 병렬 처리를 이용하여 여러개의 트리를 한번에 학습시킬 수 있음
##### 단점
- 학습시간이 상대적으로 오래걸림
#### Boot Strapping 분할
- 전체 데이터에서 중복을 허용하여 무작위로 샘플 추출
- 각 트리가 학습할 자기만의 훈련데이터 세트 생성
##### ex
- 데이터가 1000개 있다고 가정. 트리 하나는 1000개를 중복 허용해서 랜덤뽑기로 만든 데이터를 학습함
    - 어떤 데이터는 3번 포함될 수도 있음
    - 어떤 데이터는 아예 빠질 수 있음
#### Feature Bagging
- 부트스트랩으로 '행(row)'을 무작위로 추출했다면, 분할과정에서는 '열(column)'을 무작위로 추출
- 각 노드(split)를 만들 때전체 feature 중 일부만 랜덤하게 선택해서 그 중에서 최적 분할을 찾음
    - 분류문제: max_feature = sqrt(n_features)가 기본
    - 회귀문제: max_features = n_features(혹은 다른 기본값)
##### 장점
- 과적합(overfitting)을 줄임
    - 개별 트리는 과적합 될 수 있음
    - but 여러 트리가 서로 다른 방향으로 과적합하므로 평균내면 전체적으로 안정적인 예측
- 분산 감소
    - 다양한 트리들의 예측을 평균함으로써 모델의 분산이 낮아짐
- 예측 성능 증가
    - 트리 하나보다 훨씬 강력한 모델이 됨
#### scikit-learn의 랜덤 포레스트 estimator
- sklearn.ensemble.RandomForestClassifier (분류문제에서 사용)
- sklearn.ensemble.RandomForestRegressor (회귀문제에서 사용)
## XGBoost (eXtreme Gradient Boosting)
### GBM (Gradiend Boost Machine)
- Bagging 방식
    - 매번 랜덤하게 샘플을 뽑아서 독립적으로 학슴시킨 분류기들의 결과를 종합해서 앙상블 러닝 수행 - 랜덤 포레스트
- Boosting 방식
    - 매번 샘플을 뽑아서 학습시키되, **순차적으로 오차가 큰 샘플들이 뽑힐 확률이 높아지도록 가중치를 부여**하여, **다음 단계에 샘플을 뽑아서 학습시키는 알고리즘**
    - 여러 개의 약한 학습기(week learner)를 순차적으로 학습-예측하면서 잘못 예측한 데이터에 가중치 부여를 통해 오류를 개선해가면서 학습하는 방식
    - AdaBoost(Adaptive Boosing)와 GBM(Gradient Boosting Machine), XGBoost, LightGBM
#### GBM 알고리즘
- 학습과정에서 파라미터를 최적화하는데 Gradient Descent 알고리즘 사용
- 오차가 큰 샘플들이 많이 뽑히도록 할 때 Gradient Descent 알고리즘 사용
### XGBoost 장점
- 대부분의 상황에서 안정적으로 좋은 성능 발휘
- Feature Engineering을 많이 적용하지 않아도 안정적 성능을 보여줌
### XGBoost 단점
- 하이퍼 파라미터가 방대해서 튜닝하는 것이 상대적으로 어려움
### XGBoost 하이퍼 파라미터
#### 일반 파라미터 - 부스팅 수행 시 트리/선형모델 사용 결정
- booster [기본값 = gbtree]
    - gbtree: 의사결정기반모형
    - gblinear: 선형모형
    - dart: 드롭아웃 방식의 부스팅, 일부 트리를 무작위로 드롭함
- n_jobs: XGBoost를 실행하는데 사용되는 병렬 스레드 수
- verbosity [기본값 = 1]: 유효한 값은 0(무음), 1(경고), 2(정보), 3(디버그)
#### 부스터 파라미터 - 선택한 부스터에 따라 적용할 수 있는 파라미터 종류
- learning_rate [기본값: 0.3]: learning rate, learning rate가 높을수록 과적합 위험 존재
- n_estimators [기본값: 100]: 생성할 weak learner의 수
    - learning_rate가 낮을 땐, n_estimators를 충분히 늘려야 성능 확보 가능
- max_depth [기본값: 6] : 트리의 maximum depth
    - 적절한 값이 제시되어야 하고 보통 3-10 사이값이 적용됨
    - max_depth가 높을수록 모델의 복잡도가 커져 과적합되기 쉬움
#### 학습과정 파라미터 - 학습 시나리오 결정