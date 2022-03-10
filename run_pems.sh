model_path="runs/models/pemsd7-m/"

if [ ! -d "$model_path" ]; then
  mkdir "$model_path"
fi

python main.py \
    --seed 8 \
    --cuda 0 \
    --history_window 12 \
    --predict_window 9 \
    --day_slot 288 \
    --vertex 228 \
    --batch_size 32 \
    --dropout 0.5 \
    --epochs 500 \
    --spacial_kernel 3 \
    --tempora_kernel 3 \
    --model_path "$model_path" \
    --time_intvl 5 \
    --data_set "PeMS-M"
