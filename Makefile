NVCC = /usr/local/cuda-5.5/bin/nvcc

.PHONY: clean

all: plummer

clean:
	rm -f plummer_stable plummer_unstable

plummer: plummer.cu
	$(NVCC) -O3 -arch=sm_20 -o plummer_stable -DSTABLE=true plummer.cu -lm
	$(NVCC) -O3 -arch=sm_20 -o plummer_unstable plummer.cu -lm

run:
	./plummer_stable > stable.txt
	./plummer_unstable > unstable.txt
