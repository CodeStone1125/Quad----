#ifndef QUAD_HPP
#define QUAD_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <queue>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

class Model;

class Quad {
public:
    // Constructor
    Quad(Model& model, std::tuple<double, double, double, double> box, int depth);
    // Member functions
    bool is_leaf() const;
    double compute_area();
    std::vector<Quad*>  split();  // Use shared_ptr for children
    std::vector<Quad*> get_leaf_nodes(int max_depth) const;  // Use shared_ptr for leaf nodes

    // private... maybe... later...
    // Member variables
    Model* m_model;  // Use a pointer to the Model class
    std::tuple<double, double, double, double> m_box;
    bool m_leaf;
    std::vector<int> hist;
    int m_depth;
    std::tuple<int, int, int> m_color;
    double m_error;

    double m_area;
    std::vector<Quad*>  children;  // Use shared_ptr for children

    // getLastElement method for Quad
    Quad getLastElement() const {
        if (!children.empty()) {
            return *(children.back());
        }
        else {
            // If children vector is empty, return a Quad object with default values
            return Quad(*m_model, std::make_tuple(0, 0, 100, 100), m_depth);
        }
    }

    // Getter and setter for m_depth
    int getDepth() const { return m_depth; }
    void setDepth(int depth) { m_depth = depth; }

    // Getter for m_box
    std::tuple<double, double, double, double> getBox() const {
        return m_box;
    }
    // Setter for m_box
    void setBox(const std::tuple<double, double, double, double>& newBox) {
        m_box = newBox;

        // After setting the new box, you might want to recalculate related values
        // hist = calculate_histogram_cv(cropImage(m_model->im, m_box));
        // m_area = compute_area();
        // auto result = color_from_histogram(hist);
        // std::tie(m_color, m_error) = result;
    }
    double getArea() const {
        return m_area;
    }
    double getError() const {
        return m_error;
    }
    // Getter for m_color
    std::tuple<int, int, int> getColor() const {
        return m_color;
    }

    // Setter for m_color
    void setColor(const std::tuple<int, int, int>& newColor) {
        m_color = newColor;
    }
    std::vector<Quad*> getChildren() {return children; }
    // Setter for children
    void setChildren(const std::vector<Quad*>& newChildren) {children = newChildren;}
};

// // For std::priority_queue compare fountion
struct CompareQuad {
    bool operator()(const std::tuple<int, double, Quad>& a, const std::tuple<int, double, Quad>& b) const {
        // compare with 2nd element
        return std::get<1>(a) > std::get<1>(b);
    }
};

class Model {
public:
    Model(const std::string& path);
    std::vector<Quad> getQuads() const;
    double averageError() const;
    void push(Quad& quad);
    Quad pop();
    void split();
    // void render(const std::string& path, int max_depth) const;
    // private... maybe... later...
    cv::Mat im;
    int width;
    int height;
    mutable std::vector<Quad> quads_vector;
    std::priority_queue<std::tuple<int, double, Quad>, std::vector<std::tuple<int, double, Quad>>, CompareQuad> heap;
    Quad* root;  // Use a pointer to the Quad class
    double error_sum;
    // Define properties for width and height
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    Quad* getRoot() const { return root; }
    double getErrorsum(){ return error_sum; }
        // Setter for error_sum
    void setErrorSum(double newErrorSum) {
        error_sum = newErrorSum;
    }
    // Helper function to convert priority_queue to vector
    std::vector<Quad> convertPriorityQueueToVector(const std::priority_queue<std::tuple<int, double, Quad>, std::vector<std::tuple<int, double, Quad>>, CompareQuad>& pq) const {
        std::vector<Quad> vec;
        std::priority_queue<std::tuple<int, double, Quad>, std::vector<std::tuple<int, double, Quad>>, CompareQuad> temp = pq;  // Make a copy of the original priority_queue

        while (!temp.empty()) {
            vec.push_back(std::get<2>(temp.top()));  // Extract the Quad from the tuple and push it into the vector
            temp.pop();  // Remove the top element
        }

        return vec;
    }
};

#endif // QUAD_HPP
