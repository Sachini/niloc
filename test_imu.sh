#!/bin/bash

if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then
	echo "Testing script for NILoc"
	echo "Usage: \"$0 [building] [checkpoint files]\""
	echo "Buildings should be configured in niloc/config. Default options=[A, B, C]"
	exit 0
fi

echo $1 'Building'
echo $#

python niloc/cmd_test_file.py run_name=$1 dataset=$1 grid=$1 data=test task=scheduled_2branch test_cfg.test_name=out test_cfg.minimal=true ckpt_file=${2}