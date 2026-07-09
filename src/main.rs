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
use std::fs;

const OUTPUT_DIR: &str = "output";

fn output_path(file_name: &str) -> String {
    format!("{OUTPUT_DIR}/{file_name}")
}

fn main() {
    fs::create_dir_all(OUTPUT_DIR).expect("failed to create output directory");

    // Dataset: 10 exemplos, cada um com 3 features (energia, vento, ângulo)
    let dataset = Dataset::new(
        Matrix::new(
            10,
            3,
            vec![
                10.0, 80.0, 0.30, // exemplo 0
                20.0, 60.0, 0.25, // exemplo 1
                15.0, 70.0, 0.28, // exemplo 2
                25.0, 55.0, 0.22, // exemplo 3
                 8.0, 90.0, 0.35, // exemplo 4
                18.0, 65.0, 0.27, // exemplo 5
                12.0, 75.0, 0.31, // exemplo 6
                22.0, 58.0, 0.24, // exemplo 7
                 5.0, 95.0, 0.38, // exemplo 8
                28.0, 50.0, 0.20, // exemplo 9
            ],
        ),
        vec![60.0, 80.0, 70.0, 90.0, 50.0, 75.0, 65.0, 85.0, 40.0, 95.0],
    );

    println!("inputs  ({}×{}):", dataset.inputs.rows, dataset.inputs.cols);
    println!("{}", dataset.inputs);
    println!("targets: {:?}\n", dataset.targets);

    // Pipeline: 3 features → 4 neurônios (oculta) → 1 neurônio (saída)
    let mut camada1 = Layer::new(3, 4);
    let mut camada2 = Layer::new(4, 1);

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

    // guarda previsões antes do treino para o gráfico
    let preds_before: Vec<f64> = (0..z2.rows).map(|i| z2.get(i, 0)).collect();

    let lr = 0.0001;
    let epochs = 1000;

    let mut loss_history: Vec<f64> = Vec::new();

    println!("--- treino ---");
    for epoch in 0..epochs {
        // forward
        let z1 = camada1.forward(&dataset.inputs);
        let a1 = relu(&z1);
        let z2 = camada2.forward(&a1);

        // loss
        let loss = mse(&z2, &dataset.targets);
        loss_history.push(loss);
        if epoch % 100 == 0 {
            println!("epoch {epoch:>4}  loss: {loss:.4}");
        }

        // backprop
        let grad_z2 = mse_grad(&z2, &dataset.targets);
        let grads = backward(&dataset.inputs, &z1, &a1, &camada2.w, &grad_z2);

        // update W2 e b2
        for i in 0..camada2.w.rows {
            for j in 0..camada2.w.cols {
                let g = grads.grad_w2.get(i, j);
                camada2.w.set(i, j, camada2.w.get(i, j) - lr * g);
            }
        }
        for j in 0..camada2.b.len() {
            camada2.b[j] -= lr * grads.grad_b2[j];
        }

        // update W1 e b1
        for i in 0..camada1.w.rows {
            for j in 0..camada1.w.cols {
                let g = grads.grad_w1.get(i, j);
                camada1.w.set(i, j, camada1.w.get(i, j) - lr * g);
            }
        }
        for j in 0..camada1.b.len() {
            camada1.b[j] -= lr * grads.grad_b1[j];
        }
    }

    // previsão final após treino
    let z1 = camada1.forward(&dataset.inputs);
    let a1 = relu(&z1);
    let z2_final = camada2.forward(&a1);
    println!("\nprevisões após treino:\n{}", z2_final);
    println!("targets: {:?}", dataset.targets);

    let preds_after: Vec<f64> = (0..z2_final.rows).map(|i| z2_final.get(i, 0)).collect();

    plots::plot_loss(&loss_history, &output_path("03_loss.png"));
    plots::plot_predictions(
        &preds_before,
        &preds_after,
        &dataset.targets,
        &output_path("03_predictions.png"),
    );
    plots::plot_errors(
        &preds_before,
        &preds_after,
        &dataset.targets,
        &output_path("03_errors.png"),
    );
    plots::plot_activations(&output_path("03_activations.png"));

    // curva aprendida: varre energia, mantém vento=70 e ângulo=0.275 fixos
    let predict = |energia: f64| {
        let input = Matrix::new(1, 3, vec![energia, 70.0, 0.275]);
        let z1 = camada1.forward(&input);
        let a1 = relu(&z1);
        let z2 = camada2.forward(&a1);
        z2.get(0, 0)
    };
    let examples = [
        (10.0, 60.0), (20.0, 80.0), (15.0, 70.0), (25.0, 90.0),
        ( 8.0, 50.0), (18.0, 75.0), (12.0, 65.0), (22.0, 85.0),
        ( 5.0, 40.0), (28.0, 95.0),
    ];
    plots::plot_fit(&predict, &examples, (0.0, 30.0), &output_path("03_fit.png"));
}
