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
#include <cmath>

#include "nice_svm.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int argc, char* argv[]){
    auto load = false;
    auto train = true;
    auto cross = false;
    auto grid = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "load"){
            load = true;
            train = false;
        }

        if(command == "cross"){
            cross = true;
        }

        if(command == "grid"){
            grid = true;
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

    auto training_problem = svm::make_problem(dataset.training_labels, dataset.training_images, 2000);
    auto test_problem = svm::make_problem(dataset.test_labels, dataset.test_images, 0, false);

    auto mnist_parameters = svm::default_parameters();

    mnist_parameters.svm_type = C_SVC;
    mnist_parameters.kernel_type = RBF;
    mnist_parameters.probability = 1;
    mnist_parameters.C = 2.8;
    mnist_parameters.gamma = 0.0073;

    //Make it quiet
    svm::make_quiet();

    //Make sure parameters are not too messed up
    if(!svm::check(training_problem, mnist_parameters)){
        return 1;
    }

    svm::model model;

    if(load){
        model = svm::load("mnist.svm");

        if(!model){
            std::cout << "Impossible to load model" << std::endl;
        }
    }

    if(train){
        model = svm::train(training_problem, mnist_parameters);
    }

    if(grid){
        //1. Default grid
        svm::rbf_grid_search_exp(training_problem, mnist_parameters, 5);

        //2. Coarse grid (based on the values of the first search)

        svm::rbf_grid coarse_grid;
        coarse_grid.c_first = 2e-1;
        coarse_grid.c_last = 2e4;

        coarse_grid.gamma_first = 2e-9;
        coarse_grid.gamma_last = 2e-2;

        //svm::rbf_grid_search_exp(training_problem, mnist_parameters, 5, coarse_grid);

        //3. Coarser grid (based on the values of the second search)

        svm::rbf_grid coarser_grid;
        coarser_grid.c_first = 1.0;
        coarser_grid.c_last = 10.0;
        coarser_grid.c_steps = 20;

        coarser_grid.gamma_first = 2e-4;
        coarser_grid.gamma_last = 5e-2;
        coarser_grid.gamma_steps = 20;

        //svm::rbf_grid_search_exp(training_problem, mnist_parameters, 5, coarser_grid);
    }

    if(cross){
        svm::cross_validate(training_problem, mnist_parameters, 5);
    }

    std::cout << "Number of classes: " << model.classes() << std::endl;

    std::cout << "Test on training set" << std::endl;
    svm::test_model(training_problem, model);

    std::cout << "Test on test set" << std::endl;
    svm::test_model(test_problem, model);

    if(!load){
        std::cout << "Save model" << std::endl;

        if(!svm::save(model, "mnist.svm")){
            std::cout << "Unable to save model" << std::endl;
        }
    }

    return 0;
}