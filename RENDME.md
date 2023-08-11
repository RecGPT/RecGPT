Introduction
Title: RecGPT: Generative Personalized Prompts for Sequential Recommendation via ChatGPT Training Paradigm 
Datasets
Five prepared datasets are included in data folder.
Train Model
To train model on 'Beauty' dataset, change to the src folder and run following command:

For example, you can directly train Soft-gru model by running:
python main.py --data_name=Beauty --method_sequence=No --method_item=No --method_gru_theta_update=Yes
The result as follow:
