use crate::activations::relu_grad;
use crate::matrix::Matrix;

pub struct Gradients {
    pub grad_w1: Matrix,
    pub grad_b1: Vec<f64>,
    pub grad_w2: Matrix,
    pub grad_b2: Vec<f64>,
}

// Soma os gradientes de todos os exemplos do batch por coluna → gradiente do bias
fn col_sum(m: &Matrix) -> Vec<f64> {
    (0..m.cols)
        .map(|j| (0..m.rows).map(|i| m.get(i, j)).sum())
        .collect()
}

// Calcula os gradientes de W e b de cada camada, de trás pra frente
//
//   input   — X original                        (batch × in_features)
//   z1      — saída da Layer1 antes do ReLU     (batch × 4)
//   a1      — saída da Layer1 depois do ReLU    (batch × 4)
//   w2      — pesos da Layer2                   (4 × 1)
//   grad_z2 — gradiente da saída ∂L/∂z2         (batch × 1)
pub fn backward(
    input: &Matrix,
    z1: &Matrix,
    a1: &Matrix,
    w2: &Matrix,
    grad_z2: &Matrix,
) -> Gradients {
    // ── Layer 2 ───────────────────────────────────────────────────────────────
    // ∂L/∂W2 = a1.T @ grad_z2
    let grad_w2 = a1.transpose().matmul(grad_z2);

    // ∂L/∂b2 = soma de grad_z2 pelos exemplos do batch
    let grad_b2 = col_sum(grad_z2);

    // Propaga o gradiente de volta pela Layer2
    // ∂L/∂a1 = grad_z2 @ W2.T
    let grad_a1 = grad_z2.matmul(&w2.transpose());

    // ── ReLU ──────────────────────────────────────────────────────────────────
    // Propaga pelo ReLU: zera onde z1 era negativo (neurônio estava desligado)
    // ∂L/∂z1 = ∂L/∂a1 * relu'(z1)
    let grad_z1 = relu_grad(&grad_a1, z1);

    // ── Layer 1 ───────────────────────────────────────────────────────────────
    // ∂L/∂W1 = input.T @ grad_z1
    let grad_w1 = input.transpose().matmul(&grad_z1);

    // ∂L/∂b1 = soma de grad_z1 pelos exemplos do batch
    let grad_b1 = col_sum(&grad_z1);

    Gradients { grad_w1, grad_b1, grad_w2, grad_b2 }
}
