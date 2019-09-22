#ifndef CUCKOO_HPP
#define CUCKOO_HPP
//#include <pybind11/pybind11.h>
//namespace py = pybind11;

typedef unsigned long long Entry;

// TODO:// Pass this in at compile_time
const int MAX_ITERATIONS = 100;

const Entry CUCKOO_SIZE = 100;
const Entry STASH_SIZE = 10;

//template<std::size_t N = 1024>
class Cuckoo {
public:
	Cuckoo();
	~Cuckoo();
	int set();
private:
	Entry ccuckoo[CUCKOO_SIZE];
	Entry cstash[STASH_SIZE];
};

/*
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
*/
#endif
