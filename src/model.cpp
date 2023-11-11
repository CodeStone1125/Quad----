#include <pybind11/pybind11.h>

int add() {
    return 1;
}

PYBIND11_MODULE(model, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}