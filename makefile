all:
	py.test \
	 --benchmark-columns='mean,stddev,median,rounds,iterations' \
	 --benchmark-name=long \
	 --benchmark-sort=mean

test-single:
	py.test \
	 --benchmark-columns='mean,stddev,median,rounds,iterations' \
	 --benchmark-name=long \
	 --benchmark-sort=mean \
	 -m single

test-batched:
	py.test \
	 --benchmark-columns='mean,stddev,median,rounds,iterations' \
	 --benchmark-name=long \
	 --benchmark-sort=mean \
	 -m single
