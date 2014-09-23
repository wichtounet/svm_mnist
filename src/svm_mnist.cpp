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

#include "svm.h"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

void print_null(const char *s) {}

template<typename IT1, typename IT2, typename RNG>
void parallel_shuffle(IT1 first_1, IT1 last_1, IT2 first_2, IT2 last_2, RNG&& g){
    assert(std::distance(first_1, last_1) == std::distance(first_2, last_2));

    typedef typename std::iterator_traits<IT1>::difference_type diff_t;
    typedef typename std::make_unsigned<diff_t>::type udiff_t;
    typedef typename std::uniform_int_distribution<udiff_t> distr_t;
    typedef typename distr_t::param_type param_t;

    distr_t D;
    diff_t n = last_1 - first_1;

    for (diff_t i = n-1; i > 0; --i) {
        using std::swap;
        auto new_i = D(g, param_t(0, i));
        swap(first_1[i], first_1[new_i]);
        swap(first_2[i], first_2[new_i]);
    }
}

template<typename Labels, typename Images>
svm_problem make_problem(Labels& labels, Images& samples, std::size_t max = 0, bool shuffle = true){
    svm_problem problem;

    assert(labels.size() == samples.size());

    if(shuffle){
        static std::random_device rd;
        static std::mt19937_64 g(rd());

        parallel_shuffle(samples.begin(), samples.end(), labels.begin(), labels.end(), g);
    }

    if(max > 0 && max > labels.size()){
        labels.resize(max);
        samples.resize(max);
    }

    auto n_samples = labels.size();

    problem.l = n_samples;

    problem.y = new double[n_samples];
    problem.x = new svm_node*[n_samples];

    for(std::size_t s = 0; s < n_samples; ++s){
        auto features = samples[s].size();

        problem.y[s] = labels[s];
        problem.x[s] = new svm_node[features+1];

        for(std::size_t i = 0; i < features; ++i){
            problem.x[s][i].index = i+1;
            problem.x[s][i].value = samples[s][i];
        }

        //End the vector
        problem.x[s][features].index = -1;
        problem.x[s][features].value = 0.0;
    }
}

void test_model(svm_problem& problem, svm_model* model){
    double prob_estimates[10]; //TODO 10 is not fixed

    std::size_t correct = 0;

    for(std::size_t s = 0; s < problem.l; ++s){
        auto label = svm_predict_probability(model, problem.x[s], prob_estimates);

        if(label == problem.y[s]){
            ++correct;
        }
    }

    std::cout << "Samples: " << problem.l << std::endl;
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Accuracy: " << (100.0 * correct / problem.l) << "%" << std::endl;
    std::cout << "Error: " << (100.0 - (100.0 * correct / problem.l)) << "%" << std::endl;
}

void cross_validate(svm_problem& problem, svm_parameter& parameters, std::size_t n_fold){
    std::cout << "Cross validation" << std::endl;

    double *target = new double[problem.l];

    svm_cross_validation(&problem, &parameters, n_fold, target);

    std::size_t cross_correct = 0;

    for(std::size_t i = 0; i < problem.l; ++i){
        if(target[i] == problem.y[i]){
            ++cross_correct;
        }
    }

    std::cout << "Cross validation Samples: " << problem.l << std::endl;
    std::cout << "Cross validation Correct: " << cross_correct << std::endl;
    std::cout << "Cross validation Accuracy: " << (100.0 * cross_correct / problem.l) << "%" << std::endl;
    std::cout << "Cross validation Error: " << (100.0 - (100.0 * cross_correct / problem.l)) << "%" << std::endl;

    delete[] target;

    std::cout << "Cross validation done" << std::endl;
}

int main(int argc, char* argv[]){
    auto load = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "load"){
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

    auto training_problem = make_problem(training_problem, dataset.training_labels, dataset.training_images, 10000);
    auto test_problem = make_problem(test_problem, dataset.test_labels, dataset.test_images);

    svm_parameter mnist_parameters;
    mnist_parameters.svm_type = C_SVC;
    mnist_parameters.kernel_type = RBF;
    mnist_parameters.probability = 1;
    mnist_parameters.C = 2.8;
    mnist_parameters.gamma = 0.0073;

    //Default values
    mnist_parameters.degree = 3;
    mnist_parameters.coef0 = 0;
    mnist_parameters.nu = 0.5;
    mnist_parameters.cache_size = 2048;
    mnist_parameters.eps = 1e-3;
    mnist_parameters.p = 0.1;
    mnist_parameters.shrinking = 1;
    mnist_parameters.nr_weight = 0;
    mnist_parameters.weight_label = nullptr;
    mnist_parameters.weight = nullptr;

    //Make it quiet
    svm_set_print_string_function(&print_null);

    //Make sure parameters are not too messed up
    svm_check_parameter(&training_problem, &mnist_parameters);

    svm_model* model = nullptr;

    if(load){
        std::cout << "Load SVM model" << std::endl;

        model = svm_load_model("mnist.svm");

        if(!model){
            std::cout << "Impossible to load model" << std::endl;
        }

        std::cout << "SVM model loaded" << std::endl;
    } else {
        //cross_validate(training_problem, mnist_parameters, 10);

        std::cout << "Train SVM" << std::endl;

        model = svm_train(&training_problem, &mnist_parameters);

        std::cout << "Training done" << std::endl;
    }

    std::cout << "Number of classes: " << svm_get_nr_class(model) << std::endl;

    std::cout << "Test on training set" << std::endl;
    test_model(training_problem, model);

    std::cout << "Test on test set" << std::endl;
    test_model(test_problem, model);

    if(!load){
        std::cout << "Save model" << std::endl;

        if(svm_save_model("mnist.svm", model)){
            std::cout << "Unable to save model" << std::endl;
        }
    }

    std::cout << "Release data" << std::endl;

    svm_free_and_destroy_model(&model);

    //TODO Delete problem inside

    delete[] training_problem.y;
    delete[] training_problem.x;
    delete[] test_problem.y;
    delete[] test_problem.x;

    return 0;
}