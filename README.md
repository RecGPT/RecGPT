Code for our Paper 
RecGPT: Generative Personalized Prompts for Sequential Recommendation via ChatGPT Training Paradigm 

# 
NVIDIA A40; CUDA Version: 12.0


# Results
We illustrate the performance of our method comparing with different methods on four datasets, and dataset are included in data folder.

Such as Dataset Beauty
# pretrain 
```
python run_pretrain.py --data_name=Beauty --gpu_id=1 --num_hidden_layers=1 --num_attention_heads=2 --pre_epochs=300 --lr=0.001 --data_process=SASRec --hidden_size=256 --dim_feed_forward=512
```

# Prompts-tune RecGPT(X) and RecGPT(Y)

## RecGPT-1
### Run
```
python run_finetune_full.py --gpu_id=0 --data_name=Beauty --num_hidden_layers=1 --num_attention_heads=2 --hidden_size=256 --dim_feed_forward=512 --epochs=300 --Finetune_train_Prompts=Yes --Finetune_infer_GPT_recall_2=Yes --lr=0.0005 --ckp=30 --next_window_train=1
```
### RecGPT-1 result:
'H5': 0.0426, 'H10': 0.0651, 'H20': 0.0939, 'N5': 0.0283, 'N10': 0.0356, 'N20': 0.0429

## RecGPT
python run_finetune_full.py --gpu_id=0 --data_name=Beauty --num_hidden_layers=1 --num_attention_heads=2 --hidden_size=256 --dim_feed_forward=512 --epochs=300 --Finetune_train_Prompts=Yes --Finetune_infer_GPT_recall_2=Yes --lr=0.0005 --ckp=30 --next_window_train=1 --infer_GPT_recall_number_total={5,10,20} --infer_GPT_recall_number={1,...,20}
1: tune @top 5,10,20 in infer_GPT_recall_number_total,  
2: " --infer_GPT_recall_number " presents the number of of first recall item 
3: The number of the second recall is automatically obtained by the our code
### for eample："recall topk@20 and m=18,n=2 in our paper" you can run: 
```
python run_finetune_full.py --gpu_id=0 --data_name=Beauty --num_hidden_layers=1 --num_attention_heads=2 --hidden_size=256 --dim_feed_forward=512 --epochs=300 --Finetune_train_Prompts=Yes --Finetune_infer_GPT_recall_2=Yes --lr=0.0005 --ckp=30 --next_window_train=1 --infer_GPT_recall_number_total=20 --infer_GPT_recall_number=18 
```
### RecGPT result 
'H5': 0.0426, 'H10': 0.0651, 'H20': 0.0957, 'N5': 0.0283, 'N10': 0.0356, 'N20': 0.0433
### where (Top@{5,10} has the same result as RecGPT-1, while Top@20 has better result than RecGPT )
### recall topk@5 and 10, you can get the result as shown in paper only if change the number of m&n. 
