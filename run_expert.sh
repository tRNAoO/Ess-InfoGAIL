#!/bin/bash

# Train an expert policy for each mode
for i in {0..5}
do
  seed=$(echo "sacle=2; $i" | bc)
  echo 'Training expert' $i
  python train_expert.py --idx $i --seed $seed --env_id Reacher-v4 --num_modes 6 --num_steps 100000 --cuda
done
