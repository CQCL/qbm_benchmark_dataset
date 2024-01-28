#!/bin/bash
export OMP_NUM_THREADS=64
target=5
out="data/2d_hubbard"
for n in 12
do
	for beta in 0.5 1 4
	do
		for sigma in 0 0.1
		do
			for eps in 0 0.01
			do
				for model in 5
				do
					poetry run python script/benchmark.py --qre --t $target --b $beta --n $n --l $model --dn $eps --sn $sigma --output $out
					if [ $model -eq 8 ]; then
						poetry run python script/benchmark.py --qre --t $target --b $beta --n $n --l $model --dn $eps --sn $sigma --pre_l 6 --output $out
					fi
				done
			done
		done
	done
done
