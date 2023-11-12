#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quad.hpp"

const int MODE_RECTANGLE = 1;
const int MODE_ELLIPSE = 2;
const int MODE_ROUNDED_RECTANGLE = 3;

const int MODE = MODE_ELLIPSE;
const int ITERATIONS = 128;
const int LEAF_SIZE = 4;
const int PADDING = 1;
const std::tuple<int, int, int> FILL_COLOR = std::make_tuple(0, 0, 0);
const bool SAVE_FRAMES = false;
const double ERROR_RATE = 0.5;
const double AREA_POWER = 0.25;
const int OUTPUT_SCALE = 1;

// Add these constants at the beginning of your file
const int HistogramSize = 256;  // You can adjust this based on your needs
const float HistogramRange[] = {0, 256};  // You can adjust this based on your needs

// Define the comparison function
bool compare_quads(const std::tuple<bool, double, Quad*>& a, const std::tuple<bool, double, Quad*>& b) {
    return std::get<1>(a) < std::get<1>(b);
}

// ...

// Implementation of model
Model::Model(const std::string& path) : im(cv::imread(path, cv::IMREAD_COLOR)), root(*this, std::make_tuple(0, 0, im.cols, im.rows), 0), error_sum(root.getError() * root.getArea()) {
    // Other initialization code
    width = im.cols;
    height = im.rows;
    push(&root);
}

// Inside the Model class definition
void Model::push(Quad* quad) {
    double score = -quad->getError() * std::pow(quad->getArea(), AREA_POWER);
    heap.push_back(std::make_tuple(quad->is_leaf(), score, quad));
    std::push_heap(heap.begin(), heap.end(), compare_quads);  // Use the comparison function
}

std::tuple<std::tuple<int, int, int>, double> color_from_histogram(const std::vector<int>& hist) {
    // ... existing implementation ...
    return std::make_tuple(std::make_tuple(0, 0, 0), 0.0);  // Replace this line with the correct result
}

// Custom assignment operator
Quad& Quad::operator=(const Quad& other) {
    if (this != &other) {
        // Assign all non-reference members
        this->box = other.box;
        this->depth = other.depth;
        this->color = other.color;
        this->error = other.error;
        this->leaf = other.leaf;
        this->area = other.area;

        // 'model' is a reference, and we don't reassign it directly
        // Assuming 'model' is initialized in the constructor and remains unchanged
        // If it can be changed, handle that case accordingly

        // Handle 'children' separately (assuming you want to copy them)
        this->children = other.children;
    }
    return *this;
}

bool Quad::is_leaf() {
    int l, t, r, b;
    std::tie(l, t, r, b) = box;
    return r - l <= LEAF_SIZE || b - t <= LEAF_SIZE;
}

int Quad::compute_area() {
    int l, t, r, b;
    std::tie(l, t, r, b) = box;
    return (r - l) * (b - t);
}

std::vector<Quad> Quad::split() {
    int l, t, r, b;
    std::tie(l, t, r, b) = box;
    int lr = l + (r - l) / 2;
    int tb = t + (b - t) / 2;
    int new_depth = depth + 1;
    Quad tl(model, std::make_tuple(l, t, lr, tb), new_depth);
    Quad tr(model, std::make_tuple(lr, t, r, tb), new_depth);
    Quad bl(model, std::make_tuple(l, tb, lr, b), new_depth);
    Quad br(model, std::make_tuple(lr, tb, r, b), new_depth);
    children = {tl, tr, bl, br};
    return children;
}

std::vector<Quad*> Quad::get_leaf_nodes(int max_depth) {
    if (children.empty()) {
        return {this};
    }
    if (max_depth != -1 && depth >= max_depth) {
        return {this};
    }
    std::vector<Quad*> result;
    for (auto& child : children) {
        auto child_leaves = child.get_leaf_nodes(max_depth);
        result.insert(result.end(), child_leaves.begin(), child_leaves.end());
    }
    return result;
}



int add() {
    return 2;
}

namespace py = pybind11;

PYBIND11_MODULE(quad, m) {
    m.doc() = "quad module pybind11 for Model and Quad"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    py::class_<Quad>(m, "Quad")
        .def(py::init<Model&, std::tuple<int, int, int, int>, int>())
        .def("is_leaf", &Quad::is_leaf)
        .def("compute_area", &Quad::compute_area)
        .def("split", &Quad::split)
        .def("get_leaf_nodes", &Quad::get_leaf_nodes, py::arg("max_depth") = -1);

    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&>())
        .def("get_quads", &Model::get_quads)
        .def("get_average_error", &Model::get_average_error)
        .def("push", &Model::push)
        .def("pop", &Model::pop)
        .def("split", &Model::split);
}
