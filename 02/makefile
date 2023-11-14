# Объявление переменной для компилятора
CC = nvcc

# Объявление переменных для файлов
SRC = SSAA.cu
EXE = gpu
IN_DATA = in.data
OUT_DATA = out.data
FILE_NAME = test

.PHONY: all run clean
# Определение целей

run: $(EXE)
	python3 converter.py && ./$(EXE)  && python3 bytesToPng.py

$(EXE): $(SRC)
	$(CC) $< -o $@

clean:
	rm -f $(IN_DATA) $(OUT_DATA) $(EXE)

