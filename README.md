# VideoCopyDetection


## Main

### 단계별 Processing

1. **A_extract_local_features.py**: local feature 저장
2. **B_processing_clustering.py**: 클러스터 모델 생성 (cpu training)
3. **C_BOW_encoding.py**: local feature -> BOW vector
4. **D_temporal_network.py**: measure 측정

### 실행 방법

자세한 파라미터 설명은 --help 참고

#### local fingerprint 생성 커맨드 (순서대로 실행)

``` shell script
python A_extract_local_features.py  --decode_rate 1 \
                                    --group_count 5 \
                                    --cnn_model resnet50 \ 
                                    --trained True \ # 학습된 모델 사용
                                    --aggr False \ # Feature Map Max-pooling 여부
                                    --feature_path (feature 저장 경로) \
                                    --video_dataset (VCDB 데이터셋 경로)

python B_processing_clustering.py   --feature_path (local feature 경로) \
                                    --n_clust 20000 \
                                    --model_path (model 저장경로.pkl) \
                                    --n_features 1225000 # cluster 모델 생성에 사용할 local feature 개수

python C_BOW_encoding.py   --feature_path (encoding할 feature 경로) \
                           --n_clust 20000 \ 
                           --model_path (model 경로) \
                           --BOW_path (BOW vector 저장 경로)
```

### SOURCE PATH

#### Local Features 저장 경로

* /nfs_shared_/hkseok/LocalFeatures/

#### Fingerprint 저장 경로 (TFIDF 미적용)

* /nfs_shared_/hkseok/BOW/

#### Fingerprint 저장 경로 (TFIDF 적용)

* /nfs_shared_/hkseok/BOW_TFIDF/

#### 클러스터 모델 저장 경로

* /nfs_shared_/hkseok/CLUSTERS/

## simualated dataset 실험 관련

### 단계별 Processing

1. **stepA_simulated_dataset.py**: local feature 저장

2. **stepC_simulated_dataset.py**: local feature -> BOW vector

3. (a) **simulated_dataset_test.py**: 부분복사 검출 테스트

   (b) **simulated_dataset_test_full.py**: 전체복사 검출 테스트

### SOURCE PATH

#### 데이터셋 저장 경로 (변형 항목 별로 분류되어있음)

* /nfs_shared_/hkseok/SIMULATED_DATASET/videos/

#### Local Features 저장 경로

* /nfs_shared_/hkseok/LocalFeatures/multiple/simulated_dataset-1fps-res50-5sec/

#### Fingerprint 저장 경로 (TFIDF 미적용)

* /nfs_shared_/hkseok/BOW/multiple/SIMULATED_DATASET/simulated_videos/

  


## TODO

- [x] docker yml 파일수정
- [ ] Dockerfile pip install 오류 수정
- [x] requirements.txt 수정: pandas, scikit-learn 
- [x] feature, 데이터셋 경로 추가