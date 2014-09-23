//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <random>
#include <vector>
#include <cassert>

#include "nice_svm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

void print_null(const char *s) {}

int main(int argc, char* argv[]){
    auto load = false;
    auto train = true;
    auto cross = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "load"){
            load = true;
            train = false;
        }

        if(command == "cross"){
            load = true;
        }
    }

    std::cout << "Read MNIST dataset" << std::endl;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double, double>();

    if(dataset.training_images.empty() || dataset.training_labels.empty()){
        std::cout << "Impossible to read MNIST dataset" << std::endl;
        return 1;
    }

    mnist::normalize_dataset(dataset);

    std::cout << "Convert to libsvm format" << std::endl;

    auto training_problem = svm::make_problem(dataset.training_labels, dataset.training_images, 300);
    auto test_problem = svm::make_problem(dataset.test_labels, dataset.test_images, 0, false);

    auto mnist_parameters = svm::default_parameters();

    mnist_parameters.svm_type = C_SVC;
    mnist_parameters.kernel_type = RBF;
    mnist_parameters.probability = 1;
    mnist_parameters.C = 2.8;
    mnist_parameters.gamma = 0.0073;

    //Make it quiet
    svm_set_print_string_function(&print_null);

    //Make sure parameters are not too messed up
    if(!svm::check(training_problem, mnist_parameters)){
        return 1;
    }

    svm_model* model = nullptr;

    if(load){
        std::cout << "Load SVM model" << std::endl;

        model = svm_load_model("mnist.svm");

        if(!model){
            std::cout << "Impossible to load model" << std::endl;
        }

        std::cout << "SVM model loaded" << std::endl;
    }

    if(train){
        model = svm::train(training_problem, mnist_parameters);
    }

    if(cross){
        svm::cross_validate(training_problem, mnist_parameters, 10);
    }

    std::cout << "Number of classes: " << svm_get_nr_class(model) << std::endl;

    std::cout << "Test on training set" << std::endl;
    svm::test_model(training_problem, model);

    std::cout << "Test on test set" << std::endl;
    svm::test_model(test_problem, model);

    if(!load){
        std::cout << "Save model" << std::endl;

        if(svm_save_model("mnist.svm", model)){
            std::cout << "Unable to save model" << std::endl;
        }
    }

    std::cout << "Release data" << std::endl;

    svm_free_and_destroy_model(&model);

    return 0;
}