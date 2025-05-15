CC = gcc
CFLAGS = -Wall -Wextra -g -mavx2 -mfma

TARGET = program
SRC = main.c linalg_plain.c linalg_simd.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean 