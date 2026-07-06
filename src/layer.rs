use crate::matrix::Matrix;
use rand::Rng;
use std::fmt;

// ── Layer ─────────────────────────────────────────────────────────────────────
//
// Uma camada linear: output = input @ W + b
//
//   input  →  (batch × in_features)
//   W      →  (in_features × out_features)   ← um vetor de pesos por neurônio
//   b      →  (out_features)                 ← um bias por neurônio
//   output →  (batch × out_features)
//
// Cada coluna de W corresponde a um neurônio da camada.
// Com 3 entradas e 4 neurônios: W é 3×4 e b tem 4 valores.

pub struct Layer {
    pub w: Matrix,
    pub b: Vec<f64>,
}

impl Layer {
    // Inicialização com pesos aleatórios pequenos (Xavier uniform)
    // Escala por sqrt(1/in) para evitar que os valores explodam ou somem
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / in_features as f64).sqrt();

        let w_data: Vec<f64> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        Self {
            w: Matrix::new(in_features, out_features, w_data),
            b: vec![0.0; out_features],
        }
    }

    // forward: input @ W + b
    // o bias é somado a cada linha do resultado (broadcast por exemplo do batch)
    pub fn forward(&self, input: &Matrix) -> Matrix {
        let mut out = input.matmul(&self.w);
        for i in 0..out.rows {
            for j in 0..out.cols {
                out.set(i, j, out.get(i, j) + self.b[j]);
            }
        }
        out
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Layer ({}×{})", self.w.rows, self.w.cols)?;
        writeln!(f, "  W:\n{}", self.w)?;
        write!(f, "  b: {:?}", self.b)
    }
}
