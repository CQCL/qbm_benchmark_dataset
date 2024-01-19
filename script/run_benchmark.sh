#!/bin/bash
for n in 4 6 8 10
do
	for beta in 0.5 1 4 10 100
	do
		for sigma in 0 0.1 0.25
		do
			for eps in 0 0.001 0.01 0.1
			do
				for model in 0 1 6 8
				do
					poetry run python script/benchmark.py --qre --t 1 --b $beta --n $n --l $model --dn $eps --sn $sigma
					if [ $model -eq 8 ]; then
						poetry run python script/benchmark.py --qre --t 1 --b $beta --n $n --l $model --dn $eps --sn $sigma --pre_l 6
					fi
				done
			done
		done
	done
done
