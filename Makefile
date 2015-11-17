all:
	nvcc -O3 -arch=sm_50 --ptxas-options=-v -o reduction_gpu reduction_gpu.cu -std=c++11
	nvcc -O3 -arch=sm_50 --ptxas-options=-v -o reduction_gpu1 reduction_gpu1.cu -std=c++11
	nvcc -O3 -arch=sm_50 -lineinfo -Xptxas -dlcm=ca --ptxas-options=-v -o reduction_gpu2 reduction_gpu2.cu -std=c++11
	nvcc -O3 -arch=sm_50 -lineinfo -Xptxas -dlcm=ca --ptxas-options=-v -o reduction_gpu3 reduction_gpu3.cu -std=c++11
	g++ -O3 -Wall -g -o reduction_cpu reduction_cpu.cpp -std=c++11
clean:
	rm reduction_gpu reduction_gpu1 reduction_gpu2 reduction_gpu3 reduction_cpu
