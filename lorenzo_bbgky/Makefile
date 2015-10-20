CC=gcc
all: test lorenzo_bbgky.so bbgkymodule

lorenzo.bbgky.o: lorenzo_bbgky.c
	$(CC) `gsl-config --cflags` -Wall -fPIC -O -g $< -c -o $@

lorenzo_bbgky.so: lorenzo_bbgky.o
	$(CC) -shared  $<  `gsl-config --libs` -o $@

test: test_main.c lorenzo_bbgky.o
	$(CC) -o $@ $?

clean:
	rm -f test *.o *.so *.pyc
	rm -rf build
