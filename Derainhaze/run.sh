##############################################################################
# TRAIN
## raincityscape
python main_multioutput.py --mode train --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name raincityscape --learning_rate 3e-4 --batch_size 8 --num_epoch 200 --save_freq 4 --valid_freq 4
## raincityscape_pp
python main_multioutput.py --mode train --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name raincityscape_pp --learning_rate 3e-4 --batch_size 8 --num_epoch 200 --save_freq 4 --valid_freq 4
## rainhaze_synscapes
python main_multioutput.py --mode train --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name rainhaze_synscapes --learning_rate 3e-4 --batch_size 8 --num_epoch 200 --save_freq 4 --valid_freq 4
## RW2AH
python main_multioutput.py --mode train --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name RW2AH --learning_rate 2e-4 --batch_size 8 --num_epoch 500 --save_freq 4 --valid_freq 4
## Rain200H
python main_multioutput.py --mode train --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name Rain200H --learning_rate 1e-3 --batch_size 8 --patch_size 128 --num_epoch 500 --save_freq 4 --valid_freq 4
##############################################################################


##############################################################################
# TEST
## raincityscape
python main_multioutput.py --mode test --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name raincityscape --test_model results/SFNet_multioutput_wINR_lr3e-4_w1L1-0.1FFT-wAdamW/Training-Results/raincityscape/Best.pkl --save_image True
## raincityscape_pp
python main_multioutput.py --mode test --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name raincityscape_pp --test_model results/SFNet_multioutput_wINR_lr3e-4_w1L1-0.1FFT-wAdamW/Training-Results/raincityscape_pp/Best.pkl --save_image True
## rainhaze_synscapes
python main_multioutput.py --mode test --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name rainhaze_synscapes --test_model results/SFNet_multioutput_wINR_lr3e-4_w1L1-0.1FFT-wAdamW/Training-Results/rainhaze_synscapes/Best.pkl --save_image True
## RW2AH
python main_multioutput.py --mode test --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name RW2AH --test_model results/SFNet_multioutput_wINR_lr2e-4_w1L1-0.1FFT-wAdamW/Training-Results/RW2AH/Best.pkl --save_image True
## Rain200H
python main_multioutput.py --mode test --dataroot /root/workspace/d3npjrkp420c73baj4ig/datasets --task_name Rain200H --test_model results/SFNet_multioutput_wINR_lr1e-3_w1L1-0.1FFT-wDA-wMuon/Training-Results/Rain200H/Best.pkl --save_image True
##############################################################################


##############################################################################
# Evaluate (no-reference metrics)
python calculate_metrics_basicsr.py --dataset rainhaze_synscapes
python calculate_metrics_basicsr.py --dataset raincityscape
python calculate_metrics_basicsr.py --dataset raincityscape_pp
python calculate_metrics_basicsr.py --dataset Rain200H
python calculate_metrics_basicsr.py --dataset RW2AH
##############################################################################


##############################################################################
# Evaluate (no-reference metrics)
python calculate_metrics_rf.py clipiqa saved_images/Rain200H
python calculate_metrics_rf.py niqe saved_images/Rain200H
python calculate_metrics_rf.py clipiqa saved_images/RW2AH
python calculate_metrics_rf.py niqe saved_images/RW2AH
##############################################################################


##############################################################################
# FPS
python test_fps.py
##############################################################################


##############################################################################
# Params&FLOPs
python test_complexity.py
##############################################################################