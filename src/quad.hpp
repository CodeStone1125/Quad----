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
    //--------------Fountion------------//
    Quad(Model& model, std::tuple<double, double, double, double> box, int depth);
    bool is_leaf() const;
    double compute_area();
    std::vector<Quad*>  split();  // Use shared_ptr for children

    //--------------Property------------//
    Model* m_model;  // Use a pointer to the Model class
    std::tuple<double, double, double, double> m_box;
    bool m_leaf;
    std::vector<int> hist;
    int m_depth;
    std::tuple<int, int, int> m_color;
    double m_error;
    double m_area;
    std::vector<Quad*>  children;  // Use shared_ptr for children

   
    //--------------Getter------------//
    int getDepth() const { return m_depth; }
    double getArea() const { return m_area; }
    double getError() const { return m_error;}
    std::vector<Quad*> getChildren() { return children; }
    std::tuple<int, int, int> getColor() const { return m_color;}
    std::tuple<double, double, double, double> getBox() const { return m_box;}


    //--------------Setter------------//
    void setChildren(const std::vector<Quad*>& newChildren) { children = newChildren;}
    void setColor(const std::tuple<int, int, int>& newColor) { m_color = newColor;}
    void setDepth(int depth) { m_depth = depth; }

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
    //--------------Fountion------------//
    Model(const std::string& path);
    std::vector<Quad> getQuads() const;
    double averageError() const;
    void push(Quad& quad);
    Quad pop();

    //--------------Property------------//
    cv::Mat im;
    int width;
    int height;
    mutable std::vector<Quad> quads_vector;
    std::priority_queue<std::tuple<int, double, Quad>, std::vector<std::tuple<int, double, Quad>>, CompareQuad> heap;
    Quad* root;  // Use a pointer to the Quad class
    double error_sum;

    //--------------Getter------------//
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    Quad* getRoot() const { return root; }
    double getErrorsum(){ return error_sum; }

    //--------------Setter------------//
    void setErrorSum(double newErrorSum) { error_sum = newErrorSum;}

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
