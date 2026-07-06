use crate::matrix::Matrix;

// Um dataset é a combinação de:
//   inputs  → as features de cada exemplo   (Matrix: exemplos × features)
//   targets → o valor real que queremos prever (Vec: um valor por exemplo)
pub struct Dataset {
    pub inputs: Matrix,
    pub targets: Vec<f64>,
}

impl Dataset {
    pub fn new(inputs: Matrix, targets: Vec<f64>) -> Self {
        assert_eq!(
            inputs.rows,
            targets.len(),
            "número de exemplos em inputs e targets não bate"
        );
        Self { inputs, targets }
    }
}
