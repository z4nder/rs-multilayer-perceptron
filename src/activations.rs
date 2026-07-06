use crate::matrix::Matrix;

pub fn relu(input: &Matrix) -> Matrix {
    let data: Vec<f64> = (0..input.rows)
        .flat_map(|i| (0..input.cols).map(move |j| (i, j)))
        .map(|(i, j)| input.get(i, j).max(0.0))
        .collect();

    Matrix::new(input.rows, input.cols, data)
}

// Gradiente do ReLU: passa o upstream se z > 0, zera se z <= 0
// pre_activation = z1 (valores antes do ReLU, guardados no forward)
// upstream = ∂L/∂a1 (gradiente vindo da camada seguinte)
pub fn relu_grad(upstream: &Matrix, pre_activation: &Matrix) -> Matrix {
    let data: Vec<f64> = (0..upstream.rows)
        .flat_map(|i| (0..upstream.cols).map(move |j| (i, j)))
        .map(|(i, j)| {
            if pre_activation.get(i, j) > 0.0 {
                upstream.get(i, j)
            } else {
                0.0
            }
        })
        .collect();

    Matrix::new(upstream.rows, upstream.cols, data)
}
