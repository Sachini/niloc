#!/bin/bash

if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then
	echo "Training script for NILoc"
	echo "Usage: \"$0 [building]\"  : to train from scratch"
	echo "Usage: \"$0 [building] [model_checkpoint_path]\"  : to train from pre-trained checkpoint"
	echo "Buildings should be configured in niloc/config. Default options=[A, B, C]"
	exit 0
fi

echo $1 'Building'
echo $#

declare -A model_dim
model_dim["A"]=432
model_dim+=( ["B"]=704 ["C"]=672 )

if [ $# -eq 2 ]; then
  python niloc/trainer.py run_name=$1 dataset=$1 grid=$1 +arch/input@arch.encoder_input=tcn +arch/output@arch.encoder_output=cnnfc_$1 +arch/input@arch.decoder_input=cnn1d_$1 +arch/output@arch.decoder_output=cnnfc_$1 train_cfg.gpus=4 train_cfg.accelerator=ddp data.batch_size=80 arch.d_model=${model_dim[$1]} train_cfg.scheduler.monitor=val_enc_loss train_cfg.tr_ratio=0.8 train_cfg.tr_warmup=5 +train_cfg.restore_tr_ratio=False "train_cfg.load_weights_only=\"${2}\""
else
  python niloc/trainer.py run_name=$1 dataset=$1 grid=$1 +arch/input@arch.encoder_input=tcn +arch/output@arch.encoder_output=cnnfc_$1 +arch/input@arch.decoder_input=cnn1d_$1 +arch/output@arch.decoder_output=cnnfc_$1 train_cfg.gpus=4 train_cfg.accelerator=ddp data.batch_size=80 arch.d_model=${model_dim[$1]} train_cfg.scheduler.monitor=val_enc_loss
fi