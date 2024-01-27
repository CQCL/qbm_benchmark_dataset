#!/bin/bash
export OMP_NUM_THREADS=64
target=4
out="data/1d_hubbard"
for n in 4 6 8
do
	for beta in 0.5 1 4 10 100
	do
		for sigma in 0 0.1 0.25
		do
			for eps in 0 0.001 0.01 0.1
			do
				for model in 8
				do
					poetry run python benchmark.py --qre --t $target --b $beta --n $n --l $model --dn $eps --sn $sigma --output $out
					if [ $model -eq 8 ]; then
						poetry run python benchmark.py --qre --t $target --b $beta --n $n --l $model --dn $eps --sn $sigma --pre_l 6 --output $out
					fi
				done
			done
		done
	done
done
