CC=gcc
all: lorenzo_bbgky.so

bbgkymodule.o: bbgkymodule.c
	$(CC) -Wall -fPIC -O -g $< `pkg-config --cflags python` -c -o $@

lorenzo_bbgky.o: lorenzo_bbgky.c 
	$(CC) -Wall -fPIC -O -g $< -c -o $@

lorenzo_bbgky.so: lorenzo_bbgky.o bbgkymodule.o
	$(CC) -shared  $^  `pkg-config --libs python` -lblas -o $@

clean:
	rm -f *.o *.so *.pyc