#!/bin/bash
par_a=([[[12,12]]] [[[12,24]]] [[[24,24]]] [[[24,48]]] [[[24,64]]] [[[48,64]]])
par_b=(0.01)
for((i=0;i<${#par_a[@]};i++))
{
  for((j=0;j<${#par_b[@]};j++))
  {
     a=${par_a[$i]}
     b=${par_b[$j]}
     echo $a
     echo $b
     nohup python experiment_batch.py $a $b >> nohup_exp.out &
  }
}
exit 0
