first:
	nvcc -std=c++11 MMP.cu

run1:
	(python3 converter.py < 3) & (./a.out < ./tests/3/1in)  & (python3 bytesToPng.py)
run2:
	python3 converter.py < 3 &./a.out < ./tests/3/2in  & python3 bytesToPng.py
run5:
	python3 converter.py < 5 & ./a.out < ./tests/5/1in  & python3 bytesToPng.py


