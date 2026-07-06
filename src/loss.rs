use crate::matrix::Matrix;

pub fn mse(predictions: &Matrix, targets: &[f64]) -> f64 {
    let n = targets.len() as f64;
    let sum: f64 = (0..targets.len())
        .map(|i| {
            let diff = predictions.get(i, 0) - targets[i];
            diff * diff
        })
        .sum();
    sum / n
}

// Gradiente do MSE em relação à saída: (2/n) × (previsão - target)
// Retorna uma Matrix (n×1) — um valor por exemplo do batch
pub fn mse_grad(predictions: &Matrix, targets: &[f64]) -> Matrix {
    let n = targets.len() as f64;
    let data: Vec<f64> = (0..targets.len())
        .map(|i| (2.0 / n) * (predictions.get(i, 0) - targets[i]))
        .collect();
    Matrix::new(targets.len(), 1, data)
}
