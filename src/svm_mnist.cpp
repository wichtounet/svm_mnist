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

struct rbf_grid {
    double c_first = 2e-5;
    double c_last = 2e-15;
    double c_steps = 10;

    double gamma_first = 2e-15;
    double gamma_last = 2e3;
    double gamma_steps = 10;
};

//TODO Make linear version

inline void rbf_grid_search_exp(svm::problem& problem, svm_parameter& parameters, std::size_t n_fold, const rbf_grid& g = rbf_grid()){
    std::cout << "Grid Search" << std::endl;

    std::vector<double> c_values(g.c_steps);
    std::vector<double> gamma_values(g.gamma_steps);

    double c_first = g.c_first;
    double gamma_first = g.gamma_first;

    for(std::size_t i = 0; i < g.c_steps; ++i){
        c_values[i] = c_first;
        c_first *= std::pow(g.c_last / g.c_first, 1.0 / (g.c_steps - 1.0));
    }

    for(std::size_t i = 0; i < g.gamma_steps; ++i){
        gamma_values[i] = gamma_first;
        gamma_first *= std::pow(g.gamma_last / g.gamma_first, 1.0 / (g.gamma_steps - 1.0));
    }

    double max_accuracy = 0.0;
    std::size_t max_i = 0;
    std::size_t max_j = 0;

    for(std::size_t i = 0; i < g.c_steps; ++i){
        for(std::size_t j = 0; j < g.gamma_steps; ++j){
            svm_parameter new_parameter = parameters;

            new_parameter.C = c_values[i];
            new_parameter.gamma = gamma_values[j];

            auto accuracy = svm::cross_validate(problem, new_parameter, n_fold, true);

            std::cout << "C=" << c_values[i] << ",y=" << gamma_values[j] << " -> " << accuracy << std::endl;

            if(accuracy > max_accuracy){
                max_accuracy = accuracy;
                max_i = i;
                max_j = j;
            }
        }
    }

    std::cout << "Best: C=" << c_values[max_i] << ",y=" << gamma_values[max_j] << " -> " << max_accuracy << std::endl;
}

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
        //rbf_grid_search_exp(training_problem, mnist_parameters, 5);

        //2. Coarse grid (based on the values of the first search)

        rbf_grid coarse_grid;
        coarse_grid.c_first = 2e-1;
        coarse_grid.c_last = 2e4;

        coarse_grid.gamma_first = 2e-9;
        coarse_grid.gamma_last = 2e-2;

        //rbf_grid_search_exp(training_problem, mnist_parameters, 5, coarse_grid);

        //3. Coarser grid (based on the values of the second search)

        rbf_grid coarser_grid;
        coarser_grid.c_first = 1.0;
        coarser_grid.c_last = 10.0;
        coarser_grid.c_steps = 20;

        coarser_grid.gamma_first = 2e-4;
        coarser_grid.gamma_last = 5e-2;
        coarser_grid.gamma_steps = 20;

        rbf_grid_search_exp(training_problem, mnist_parameters, 5, coarser_grid);
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