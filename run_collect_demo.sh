#!/bin/bash

# collect unlabeled demonstrations
for i in {0..5}
do
  echo 'Collecting unlabeled demo' $i
  python collect_demo.py --idx $i --seed 0 --env_id Reacher-v4 --labeled 0 --num_modes 6 --buffer_size 100000 --std 0.1
done

# collect labeled demonstrations, 0.5% of the unlabeled demonstrations in Reacher-v4
for i in {0..5}
do
  echo 'Collecting labeled demo' $i
  python collect_demo.py --idx $i --seed 1 --env_id Reacher-v4 --labeled 1 --num_modes 6 --buffer_size 500 --std 0.1
done

