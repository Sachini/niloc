#!/bin/bash

if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then
	echo "Pre-training script for NILoc"
	echo "Usage: $0 [building]"
	echo "Buildings should be configured in niloc/config. Default options=[A, B, C]"
	exit 0
fi

echo $1 'Building'

declare -A model_dim
model_dim["A"]=432
model_dim+=( ["B"]=704 ["C"]=672 )


python niloc/trainer.py run_name=$1_syn dataset=$1_syn grid=$1 +arch/input@arch.encoder_input=tcn +arch/output@arch.encoder_output=cnnfc_$1 +arch/input@arch.decoder_input=cnn1d_$1 +arch/output@arch.decoder_output=cnnfc_$1 train_cfg.gpus=4 train_cfg.accelerator=ddp data.batch_size=80 arch.d_model=${model_dim[$1]} train_cfg.scheduler.monitor=val_enc_loss