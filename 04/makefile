# Имя компилятора
CXX := g++
# Флаги компиляции
CXXFLAGS := -std=c++11 -Wall -Wextra -fsanitize=address

# Имя исполняемого файла
TARGET := cpu_program

# Список исходных файлов
SOURCES := cpu.cpp 

# Генерация имен объектных файлов из исходных файлов
OBJECTS := $(SOURCES:.cpp=.o)

# Правило по умолчанию
all: $(TARGET)

# Правило сборки исполняемого файла из объектных файлов
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Правило компиляции каждого cpp-файла в объектный файл
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Правило очистки (удаление объектных файлов и исполняемого файла)
clean:
	rm -f $(OBJECTS) $(TARGET)
