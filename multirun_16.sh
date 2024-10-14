# bash scripts/run_train.sh configs/k400/k400_train_video_vitl-14-f8.yaml

# bash scripts/run_train.sh configs/k400/zmh_k400_train_video_vitb-16-f8.yaml

bash scripts/run_test_zeroshot.sh configs/hmdb51/hmdb_split1.yaml MoTE_B16.pt --test_clips 3

# bash scripts/run_test.sh configs/k400/zmh_k400_train_video_vitb-16-f8.yaml MoTE_B16.pt

