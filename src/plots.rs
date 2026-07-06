use plotters::prelude::*;
use std::fs;
use std::path::Path;

const BG: RGBColor = RGBColor(8, 6, 24);
const PANEL_BG: RGBColor = RGBColor(20, 10, 46);
const GRID: RGBColor = RGBColor(92, 46, 160);
const TEXT: RGBColor = RGBColor(244, 239, 231);
const PURPLE: RGBColor = RGBColor(164, 82, 255);
const GREEN_NEON: RGBColor = RGBColor(181, 223, 0);

// Arquitetura fixa: 1 entrada → 4 neurônios ocultos → 1 saída
// Pesos determinísticos para o gráfico ser reproduzível

// W1: peso de cada neurônio oculto (1 entrada × 4 neurônios)
const W1: [f64; 4] = [1.5, -2.0, 1.0, -1.0];
// B1: bias de cada neurônio oculto
const B1: [f64; 4] = [-2.0, 1.5, -0.5, 0.5];
// W2: peso da saída (4 neurônios → 1 saída)
const W2: [f64; 4] = [0.5, -0.3, 0.8, -0.6];

// Sem ativação — tudo linear, colapsa numa reta
fn forward_linear(x: f64) -> f64 {
    let hidden: Vec<f64> = W1.iter().zip(B1.iter()).map(|(w, b)| x * w + b).collect();
    hidden.iter().zip(W2.iter()).map(|(h, w)| h * w).sum::<f64>()
}

// Com ReLU — cada neurônio zera se negativo, criando dobras
fn forward_relu(x: f64) -> f64 {
    let hidden: Vec<f64> = W1
        .iter()
        .zip(B1.iter())
        .map(|(w, b)| (x * w + b).max(0.0))
        .collect();
    hidden.iter().zip(W2.iter()).map(|(h, w)| h * w).sum::<f64>()
}

pub fn plot_activations(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).unwrap();
        }
    }

    let root = BitMapBackend::new(path, (1400, 550)).into_drawing_area();
    root.fill(&BG).unwrap();
    let (left, right) = root.split_horizontally(700);

    let xs: Vec<f64> = (0..=600).map(|i| -3.0 + i as f64 * 6.0 / 600.0).collect();

    let linear_vals: Vec<f64> = xs.iter().map(|&x| forward_linear(x)).collect();
    let relu_vals: Vec<f64> = xs.iter().map(|&x| forward_relu(x)).collect();

    // range compartilhado entre os dois painéis para facilitar a comparação
    let all_vals: Vec<f64> = linear_vals.iter().chain(relu_vals.iter()).cloned().collect();
    let y_min = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min) * 0.12;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    // ── painel esquerdo: sem ativação ────────────────────────────────────────
    left.fill(&PANEL_BG).unwrap();

    let mut chart_l = ChartBuilder::on(&left)
        .caption(
            "Sem ativação — layers colapsam em 1 reta",
            ("sans-serif", 15).into_font().color(&TEXT),
        )
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(-3f64..3f64, y_lo..y_hi)
        .unwrap();

    chart_l
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc("input x")
        .y_desc("output y")
        .draw()
        .unwrap();

    // linha zero de referência
    chart_l
        .draw_series(LineSeries::new(
            vec![(-3.0, 0.0), (3.0, 0.0)],
            TEXT.mix(0.2).stroke_width(1),
        ))
        .unwrap();

    let linear_series: Vec<(f64, f64)> = xs
        .iter()
        .zip(linear_vals.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    chart_l
        .draw_series(LineSeries::new(linear_series, PURPLE.stroke_width(3)))
        .unwrap()
        .label("Linear → Linear(x)  — sempre reta")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.stroke_width(3))
        });

    chart_l
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(TEXT.mix(0.35))
        .background_style(BG.mix(0.9))
        .label_font(("sans-serif", 12).into_font().color(&TEXT))
        .draw()
        .unwrap();

    // ── painel direito: com ReLU ──────────────────────────────────────────────
    right.fill(&PANEL_BG).unwrap();

    let mut chart_r = ChartBuilder::on(&right)
        .caption(
            "Com ReLU — dobras, não colapsa",
            ("sans-serif", 15).into_font().color(&TEXT),
        )
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(-3f64..3f64, y_lo..y_hi)
        .unwrap();

    chart_r
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc("input x")
        .y_desc("output y")
        .draw()
        .unwrap();

    // linha zero de referência
    chart_r
        .draw_series(LineSeries::new(
            vec![(-3.0, 0.0), (3.0, 0.0)],
            TEXT.mix(0.2).stroke_width(1),
        ))
        .unwrap();

    let relu_series: Vec<(f64, f64)> = xs
        .iter()
        .zip(relu_vals.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    chart_r
        .draw_series(LineSeries::new(relu_series, GREEN_NEON.stroke_width(3)))
        .unwrap()
        .label("Linear → ReLU → Linear(x)  — dobras")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 24, y)], GREEN_NEON.stroke_width(3))
        });

    chart_r
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(TEXT.mix(0.35))
        .background_style(BG.mix(0.9))
        .label_font(("sans-serif", 12).into_font().color(&TEXT))
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Salvo em {path}");
}
