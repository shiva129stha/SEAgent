software=vs_code
export json_file="your data after to_sft.py"
accelerate launch --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/sft_gui.py \
                  --config src/r1-v/configs/ui_tars_7b_vscode.yaml --output_dir data/ui_tars_7b_${software}_p0_ls30_e4 \
                  --num_train_epochs 4
