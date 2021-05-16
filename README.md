# MoE-ASD
A Mixture-of-Experts Model for Antonym-Synonym Discrimination
%%%run our method
> cd model

if you want combine train stage and test stage and run multiple times, please use train_model.py, if you need change other parameters, please enter code
e.g.  
> python train_model.py --pos 'adj' --expert_size 256 --embed_size 300 --projection_size 4

if you want train one time and save the model, please use train.py
e.g.  
>python train.py --pos 'adj' --expert_size 256 --embed_size 300 --projection_size 4 --model_file './model_adj.pkl'

if you want use ready-made model, please use eval.py
e.g.  
>python eval.py --pos 'adj' --expert_size 256 --embed_size 300 --projection_size 4 --model_file './model_adj.pkl'

tips:
the dlCE embedding size is 100


%%% run baseline
> cd baseline
> python baseline_train.py --pos 'adj' --embed 'fasttext'
