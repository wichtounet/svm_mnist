//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "svm.h"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

constexpr const std::size_t n_samples = 5000;

int main(int argc, char* argv[]){
    std::cout << "Read MNIST dataset" << std::endl;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double, double>(n_samples);

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read MNIST dataset" << std::endl;
        return 1;
    }

//    mnist::normalize_dataset(dataset);

    std::cout << "Convert to libsvm format" << std::endl;

    svm_problem mnist_problem;

    mnist_problem.l = n_samples;
    mnist_problem.y = &dataset.training_labels[0];

    mnist_problem.x = new svm_node*[n_samples];

    for(std::size_t s = 0; s < n_samples; ++s){
        mnist_problem.x[s] = new svm_node[784];

        for(std::size_t i = 0; i < 784; ++i){
            mnist_problem.x[s][i].index = i+1;
            mnist_problem.x[s][i].value = dataset.training_images[s][i];
        }
    }

    svm_parameter mnist_parameters;
    mnist_parameters.svm_type = C_SVC;
    mnist_parameters.kernel_type = RBF;
    mnist_parameters.probability = 1;

    //Default values
    mnist_parameters.degree = 3;
    mnist_parameters.gamma = 0;
    mnist_parameters.coef0 = 0;
    mnist_parameters.nu = 0.5;
    mnist_parameters.cache_size = 100;
    mnist_parameters.C = 1;
    mnist_parameters.eps = 1e-3;
    mnist_parameters.p = 0.1;
    mnist_parameters.shrinking = 1;
    mnist_parameters.probability = 0;
    mnist_parameters.nr_weight = 0;
    mnist_parameters.weight_label = NULL;
    mnist_parameters.weight = NULL;

    svm_check_parameter(&mnist_problem, &mnist_parameters);

    std::cout << "Train SVM" << std::endl;

    auto model = svm_train(&mnist_problem, &mnist_parameters);

    std::cout << "Release data" << std::endl;

    for(std::size_t s = 0; s < n_samples; ++s){
        delete[] mnist_problem.x[s];
    }

    delete[] mnist_problem.x;

    return 0;
}