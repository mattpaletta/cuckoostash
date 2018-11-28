LIB_DIR = cuckoo/gpu/lib

default: pycuckoo

pycuckoo: setup.py cuckoo/gpu/pycuckoo.pyx $(LIB_DIR)/libcuckoo.a
	python3 setup.py build_ext --inplace

$(LIB_DIR)/libcuckoo.a:
	make -C $(LIB_DIR) libcuckoo.a

clean:
	make -C $(LIB_DIR) clean
	rm -f *.so
	rm -rf build/
