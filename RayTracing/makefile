PROGRAM = kp
DEBUGFILE = deb
COMPILER = nvcc
# COMPILER = /usr/local/cuda/bin/nvcc

OPTIONS = -std=c++11 -O2
$(PROGRAM): main.cu
	$(COMPILER) main.cu -o $(PROGRAM) $(OPTIONS) 

TOPNG = topng
$(TOPNG): PNG_from_binary.cpp
	g++ PNG_from_binary.cpp -std=c++11 -O2 -o $(TOPNG)

run_cpu: $(PROGRAM) $(TOPNG)
	time bash run.sh $(PROGRAM) --cpu $(TOPNG)

run_gpu: $(PROGRAM) $(TOPNG)
	time bash run.sh $(PROGRAM) --gpu $(TOPNG)

debug: main.cu $(TOPNG)
	$(COMPILER) -Ddebug main.cu -o $(DEBUGFILE) -g
	time bash debugrun.sh $(DEBUGFILE) in/debug.in 

clean:
	rm $(TOPNG) $(PROGRAM) $(DEBUGFILE) -r png -r res -r bin o.plt