all: test lorenzo_bbgky.so bbgkymodule

lorenzo_bbgky.so: lorenzo_bbgky.c
	$(CC) -Wall -g -fPIC -shared -o $@ $? -lc

test: test_main.c lorenzo_bbgky.o
	$(CC) -o $@ $?

clean:
	rm -f test *.o *.so *.pyc
	rm -rf build
