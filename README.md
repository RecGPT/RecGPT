
Code for our Paper 
RecGPT: Generative Personalized Prompts for Sequential Recommendation via ChatGPT Training Paradigm 

## Results
We illustrate the performance of our method comparing with different methods on four datasets, and dataset are included in data folder.


## Pre-train 
```shell script
python run_pretrain.py --data_name=Beauty --gpu_id=1 --num_hidden_layers=1 --num_attention_heads=2 --pre_epochs=300 --lr=0.001 --data_process=SASRec --hidden_size=256 --dim_feed_forward=512
```

## Prompts-tune

```shell script
python run_finetune_full.py --gpu_id=0 --data_name=Beauty --num_hidden_layers=1 --num_attention_heads=2 --hidden_size=256 --dim_feed_forward=512 --epochs=300 --Finetune_train_Prompts=Yes --Finetune_infer_GPT_recall_2=Yes --lr=0.0005 --ckp=30 --next_window_train=1 --infer_GPT_recall_number_total=20 --infer_GPT_recall_number=19
```

