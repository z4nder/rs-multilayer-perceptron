mod activations;
mod backward;
mod dataset;
mod layer;
mod loss;
mod matrix;
mod plots;

use activations::relu;
use backward::backward;
use dataset::Dataset;
use layer::Layer;
use loss::{mse, mse_grad};
use matrix::Matrix;

fn main() {
    // Dataset: 2 exemplos, cada um com 3 features (energia, peso, aero)
    let dataset = Dataset::new(
        Matrix::new(
            2,
            3,
            vec![
                10.0, 80.0, 0.30, // exemplo 0
                20.0, 60.0, 0.25, // exemplo 1
            ],
        ),
        vec![60.0, 80.0], // distâncias reais (targets)
    );

    println!("inputs  ({}×{}):", dataset.inputs.rows, dataset.inputs.cols);
    println!("{}", dataset.inputs);
    println!("targets: {:?}\n", dataset.targets);

    // Pipeline: 3 features → 4 neurônios (oculta) → 1 neurônio (saída)
    let camada1 = Layer::new(3, 4);
    let camada2 = Layer::new(4, 1);

    println!("camada1 {}\n", camada1);
    println!("camada2 {}\n", camada2);

    // forward com ReLU entre as camadas
    let z1 = camada1.forward(&dataset.inputs); // X @ W1 + b1
    let a1 = relu(&z1);                        // ReLU — zera negativos
    let z2 = camada2.forward(&a1);             // a1 @ W2 + b2 — saída final

    println!("z1 (antes do ReLU):\n{}", z1);
    println!("a1 (depois do ReLU):\n{}", a1);
    println!("previsões (saída final):\n{}", z2);

    let loss = mse(&z2, &dataset.targets);
    println!("loss (MSE): {:.4}", loss);

    let grad_z2 = mse_grad(&z2, &dataset.targets);
    println!("\ngradiente da saída ∂L/∂z2 ({}×{}):\n{}", grad_z2.rows, grad_z2.cols, grad_z2);

    let grads = backward(&dataset.inputs, &z1, &a1, &camada2.w, &grad_z2);
    println!("∂L/∂W2 ({}×{}):\n{}", grads.grad_w2.rows, grads.grad_w2.cols, grads.grad_w2);
    println!("∂L/∂b2: {:?}\n", grads.grad_b2);
    println!("∂L/∂W1 ({}×{}):\n{}", grads.grad_w1.rows, grads.grad_w1.cols, grads.grad_w1);
    println!("∂L/∂b1: {:?}", grads.grad_b1);

    plots::plot_activations("output/03_activations.png");
}
