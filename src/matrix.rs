use std::fmt;

// ── Vec2 ──────────────────────────────────────────────────────────────────────
// Vetor 2D simples — reservado para intuição geométrica futura

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

#[allow(dead_code)]
impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn add(&self, other: &Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    pub fn scale(&self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }

    // produto escalar: a · b = Σ(aᵢ * bᵢ)
    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

// ── Matrix ────────────────────────────────────────────────────────────────────
// Matriz genérica rows×cols armazenada em row-major order

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data.len() != rows * cols");
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, vec![0.0; rows * cols])
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }

    // transposta: (m×n) → (n×m)
    // necessária no backprop: grad_W = input.T @ grad_out
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    // multiplicação matricial: (m×k) · (k×n) → (m×n)
    // cada elemento C[i][j] = Σₖ A[i][k] * B[k][j]
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "dimensões incompatíveis: {}×{} · {}×{}",
            self.rows, self.cols, other.rows, other.cols
        );

        let m = self.rows;
        let n = other.cols;
        let k = self.cols;
        let mut result = Matrix::zeros(m, n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.get(i, p) * other.get(p, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:7.3}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
