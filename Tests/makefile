
make:
	nvcc example.cu -o gpu
run: make
	python3 converter.py && ./gpu < test && python3 bytesToPng.py
clean:
	rm in.data out.data gpu
