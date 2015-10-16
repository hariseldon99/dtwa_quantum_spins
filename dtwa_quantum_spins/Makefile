all: test libtest.so testmodule

libtest.so: libtest.c
	$(CC) -Wall -g -fPIC -shared -o $@ $? -lc

test: test_main.c libtest.o
	$(CC) -o $@ $?

testmodule: testmodule.c
	python setup.py build

clean:
	rm -f test libtest.o libtest.so *.pyc
	rm -rf build
