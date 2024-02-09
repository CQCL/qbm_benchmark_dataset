#!/bin/bash
export OMP_NUM_THREADS=8
target=0
out="data/1d_tfim"
for n in 4 6 8
do
	for beta in 0.5 1 2 4
	do
		for sn in 0 0.1
		do
			for dn in 0 0.001
			do
				for model in 8 9
				do
					poetry run python script/benchmark.py --qre --t $target --b $beta --n $n --l $model --dn $dn --sn $sn --output $out
					# if [ $model -eq 8 -a $sn -eq 0 -a $dn -eq 0 ]; then
					# 	poetry run python script/benchmark.py --qre --t $target --b $beta --n $n --l $model --pre_l 6 --output $out
					# fi
				done
			done
		done
	done
done
