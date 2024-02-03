#!/bin/bash
export OMP_NUM_THREADS=64
target=0
out="data/1d_tfim"
for n in 4 6 8 10 12
do
	for beta in 0.5 1 2 4
	do
		for sn in 0
		do
			for dn in 0
			do
				for model in 0
				do
					poetry run python script/benchmark.py --qre --t $target --b $beta --n $n --l $model --lr 0.5 --dn $dn --sn $sn --output $out
					# if [ $model -eq 8 ]; then
					# 	poetry run python script/benchmark.py --qre --t $target --b $beta --n $n --l $model --pre_l 6 --output $out
					# fi
				done
			done
		done
	done
done
