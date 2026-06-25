#pragma once
#include "../algebra/tensor_ops.h"
#include <stdexcept>

namespace utec::tf::optimizers {

    struct SGD {
        float learning_rate;

        explicit SGD(float learning_rate) : learning_rate(learning_rate) {
            if (learning_rate <= 0.0f) {
                throw std::invalid_argument("el learning rate debe ser positivo");
            }
        }

        void update(utec::Tensor<float>& parameter, const utec::Tensor<float>& gradient) {
            if (!(parameter.shape() == gradient.shape())) {
                throw std::invalid_argument("el parametro y el gradiente deben tener la misma forma");
            }
            for (size_t i = 0; i < parameter.size(); i++) {
                parameter.flat(i) = parameter.flat(i) - learning_rate * gradient.flat(i);
            }
        }
    };

}

using namespace utec::tf;