# VideoCopyDetection

## 단계별 Processing
1. A_extract_local_features.py: local feature 저장
2. B_processing_clustering.py: 클러스터 모델 생성 (cpu training)
3. C_BOW_encoding.py: local feature -> BOW vector
4. D_temporal_network.py: measure 측정

## 실행 방법
자세한 파라미터 설명은 --help 참고
### multi-keyframe local fingerprint (sum all local features of (group_count) frames)
``` shell script
python A_extract_local_features.py  --decode_rate 1 \
                                    --group_count 5 \
                                    --cnn_model resnet50 \ 
                                    --trained True \ # 학습된 모델 사용
                                    --aggr False \
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

### multi-keyframe local fingerprint (maxpooling)
``` shell script
python A_extract_local_features.py  --decode_rate 1 \
                                    --group_count 5 \
                                    --cnn_model resnet50 \ 
                                    --trained True \ # 학습된 모델 사용
                                    --aggr True \
                                    --feature_path (feature 저장 경로) \
                                    --video_dataset (VCDB 데이터셋 경로)

이하 동일
```