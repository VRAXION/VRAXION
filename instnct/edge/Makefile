# INSTNCT Edge — Build for desktop test / RPi / Arduino
CC = gcc
CFLAGS = -O2 -Wall -std=c99

# Desktop test build
all: instnct_test

# Generate LUT and network data
instnct_lut.c: gen_lut.py
	python gen_lut.py > instnct_lut.c

instnct_data.c: export_network.py
	python export_network.py > instnct_data.c

# Build
instnct_test: instnct_edge.c instnct_main.c instnct_lut.c instnct_data.c
	$(CC) $(CFLAGS) -o instnct_test instnct_edge.c instnct_main.c instnct_lut.c instnct_data.c -lm

# Run test: pipe text in, get predictions out
test: instnct_test
	echo "Hello World" | ./instnct_test

# Size report
size: instnct_test
	size instnct_test
	@echo "---"
	@echo "Data sizes:"
	@wc -c instnct_data.c instnct_lut.c

clean:
	rm -f instnct_test instnct_lut.c instnct_data.c

.PHONY: all test size clean
