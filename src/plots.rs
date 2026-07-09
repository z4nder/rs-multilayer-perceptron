use plotters::prelude::*;

const BG: RGBColor = RGBColor(8, 6, 24);
const PANEL_BG: RGBColor = RGBColor(20, 10, 46);
const GRID: RGBColor = RGBColor(92, 46, 160);
const TEXT: RGBColor = RGBColor(244, 239, 231);
const PURPLE: RGBColor = RGBColor(164, 82, 255);
const GREEN_NEON: RGBColor = RGBColor(181, 223, 0);
const ORANGE_FIRE: RGBColor = RGBColor(255, 120, 0);
const GOLD: RGBColor = RGBColor(255, 180, 50);

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

// Varre a feature energia (coluna 0) de x_min a x_max,
// mantendo vento e ângulo fixos, e plota a curva da rede treinada.
// Os exemplos reais aparecem como pontos com linha de erro para o target.
pub fn plot_fit(
    predict_fn: impl Fn(f64) -> f64,
    examples: &[(f64, f64)], // (energia, target) dos exemplos reais
    x_range: (f64, f64),
    path: &str,
) {
    let root = BitMapBackend::new(path, (900, 500)).into_drawing_area();
    root.fill(&BG).unwrap();
    root.fill(&PANEL_BG).unwrap();

    let steps = 300;
    let curve: Vec<(f64, f64)> = (0..=steps)
        .map(|i| {
            let x = x_range.0 + i as f64 * (x_range.1 - x_range.0) / steps as f64;
            (x, predict_fn(x))
        })
        .collect();

    let all_y: Vec<f64> = curve.iter().map(|(_, y)| *y)
        .chain(examples.iter().map(|(_, t)| *t))
        .collect();
    let y_min = all_y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pad = (y_max - y_min) * 0.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Curva aprendida — energia vs distância (vento e ângulo fixos)",
            ("sans-serif", 15).into_font().color(&TEXT),
        )
        .margin(30)
        .x_label_area_size(42)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.0..x_range.1, (y_min - pad)..(y_max + pad))
        .unwrap();

    chart
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc("energia")
        .y_desc("distância prevista")
        .draw()
        .unwrap();

    // curva da rede
    chart
        .draw_series(LineSeries::new(curve, PURPLE.stroke_width(3)))
        .unwrap()
        .label("previsão da rede")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.stroke_width(3)));

    // linha de erro e pontos dos exemplos reais
    for &(x, target) in examples {
        let pred = predict_fn(x);
        chart
            .draw_series(LineSeries::new(
                vec![(x, pred), (x, target)],
                ORANGE_FIRE.mix(0.8).stroke_width(2),
            ))
            .unwrap();

        let mid = (pred + target) / 2.0;
        let err = pred - target;
        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:+.1}", err),
                (x + 0.3, mid),
                ("sans-serif", 11).into_font().color(&ORANGE_FIRE),
            )))
            .unwrap();
    }

    // pontos dos targets reais
    chart
        .draw_series(examples.iter().map(|&(x, t)| Circle::new((x, t), 8, GREEN_NEON.filled())))
        .unwrap()
        .label("target real")
        .legend(|(x, y)| Circle::new((x + 8, y), 6, GREEN_NEON.filled()));

    // pontos das previsões nos exemplos
    chart
        .draw_series(examples.iter().map(|&(x, _)| {
            Circle::new((x, predict_fn(x)), 7, GOLD.filled())
        }))
        .unwrap()
        .label("previsão nos exemplos")
        .legend(|(x, y)| Circle::new((x + 8, y), 6, GOLD.filled()));

    chart
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

pub fn plot_errors(
    before: &[f64],
    after: &[f64],
    targets: &[f64],
    path: &str,
) {
    let n = targets.len();
    let root = BitMapBackend::new(path, (1000, 480)).into_drawing_area();
    root.fill(&BG).unwrap();
    let (left, right) = root.split_horizontally(500);

    let all_vals: Vec<f64> = targets.iter().chain(before.iter()).chain(after.iter()).cloned().collect();
    let y_min = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pad = (y_max - y_min) * 0.2;
    let y_lo = (y_min - pad).min(-pad.abs());
    let y_hi = y_max + pad;

    let draw_panel = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                      preds: &[f64],
                      title: &str,
                      pred_color: RGBColor| {
        area.fill(&PANEL_BG).unwrap();

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 14).into_font().color(&TEXT))
            .margin(28)
            .x_label_area_size(36)
            .y_label_area_size(60)
            .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), y_lo..y_hi)
            .unwrap();

        chart
            .configure_mesh()
            .bold_line_style(GRID.mix(0.45))
            .light_line_style(GRID.mix(0.18))
            .axis_style(TEXT.mix(0.75))
            .label_style(("sans-serif", 12).into_font().color(&TEXT))
            .x_desc("exemplo")
            .y_desc("valor")
            .draw()
            .unwrap();

        // linha de erro entre previsão e target
        for (i, (&p, &t)) in preds.iter().zip(targets.iter()).enumerate() {
            chart
                .draw_series(LineSeries::new(
                    vec![(i as f64, p), (i as f64, t)],
                    ORANGE_FIRE.mix(0.7).stroke_width(2),
                ))
                .unwrap();

            // valor do erro no meio da linha
            let mid = (p + t) / 2.0;
            let err = p - t;
            chart
                .draw_series(std::iter::once(Text::new(
                    format!("{:+.1}", err),
                    (i as f64 + 0.06, mid),
                    ("sans-serif", 11).into_font().color(&ORANGE_FIRE),
                )))
                .unwrap();
        }

        // pontos de target
        chart
            .draw_series(targets.iter().enumerate().map(|(i, &t)| {
                Circle::new((i as f64, t), 8, GREEN_NEON.filled())
            }))
            .unwrap()
            .label("target")
            .legend(|(x, y)| Circle::new((x + 8, y), 6, GREEN_NEON.filled()));

        // pontos de previsão
        chart
            .draw_series(preds.iter().enumerate().map(|(i, &p)| {
                Circle::new((i as f64, p), 8, pred_color.filled())
            }))
            .unwrap()
            .label("previsão")
            .legend(move |(x, y)| Circle::new((x + 8, y), 6, pred_color.filled()));

        // label do erro na legenda
        chart
            .draw_series(std::iter::once(Circle::new((0.0, -9999.0), 0, TRANSPARENT)))
            .unwrap()
            .label("erro")
            .legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 16, y)], ORANGE_FIRE.mix(0.7).stroke_width(2))
            });

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .border_style(TEXT.mix(0.35))
            .background_style(BG.mix(0.9))
            .label_font(("sans-serif", 11).into_font().color(&TEXT))
            .draw()
            .unwrap();
    };

    draw_panel(&left, before, "Antes do treino", PURPLE);
    draw_panel(&right, after, "Depois do treino", GOLD);

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_loss(history: &[f64], path: &str) {
    let root = BitMapBackend::new(path, (900, 500)).into_drawing_area();
    root.fill(&BG).unwrap();
    root.fill(&PANEL_BG).unwrap();

    let max_loss = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n = history.len() as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss ao longo do treino", ("sans-serif", 16).into_font().color(&TEXT))
        .margin(30)
        .x_label_area_size(42)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..n, 0f64..max_loss * 1.1)
        .unwrap();

    chart
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc("epoch")
        .y_desc("loss (MSE)")
        .draw()
        .unwrap();

    let series: Vec<(f64, f64)> = history.iter().enumerate().map(|(i, &l)| (i as f64, l)).collect();

    let final_loss = history.last().copied().unwrap_or(0.0);

    chart
        .draw_series(LineSeries::new(series, PURPLE.stroke_width(3)))
        .unwrap()
        .label(format!("loss final: {:.4}", final_loss))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.stroke_width(3)));

    chart
        .draw_series(std::iter::once(Circle::new(
            ((history.len() - 1) as f64, final_loss),
            6,
            GREEN_NEON.filled(),
        )))
        .unwrap();

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(TEXT.mix(0.35))
        .background_style(BG.mix(0.9))
        .label_font(("sans-serif", 12).into_font().color(&TEXT))
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_predictions(
    before: &[f64],
    after: &[f64],
    targets: &[f64],
    path: &str,
) {
    let n = targets.len();
    let root = BitMapBackend::new(path, (900, 500)).into_drawing_area();
    root.fill(&BG).unwrap();
    let (left, right) = root.split_horizontally(450);

    let max_val = targets
        .iter()
        .chain(before.iter())
        .chain(after.iter())
        .cloned()
        .fold(0f64, f64::max)
        * 1.2;

    let draw_panel = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                      preds: &[f64],
                      title: &str,
                      bar_color: RGBColor| {
        area.fill(&PANEL_BG).unwrap();

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 14).into_font().color(&TEXT))
            .margin(24)
            .x_label_area_size(36)
            .y_label_area_size(52)
            .build_cartesian_2d(0f64..(n as f64 + 0.5), 0f64..max_val)
            .unwrap();

        chart
            .configure_mesh()
            .bold_line_style(GRID.mix(0.45))
            .light_line_style(GRID.mix(0.18))
            .axis_style(TEXT.mix(0.75))
            .label_style(("sans-serif", 12).into_font().color(&TEXT))
            .x_desc("exemplo")
            .y_desc("valor")
            .draw()
            .unwrap();

        // barras de target — com label para legenda
        chart
            .draw_series(targets.iter().enumerate().map(|(i, &t)| {
                let x = i as f64 + 0.6;
                Rectangle::new([(x, 0.0), (x + 0.25, t)], GREEN_NEON.mix(0.5).filled())
            }))
            .unwrap()
            .label("target")
            .legend(|(x, y)| {
                Rectangle::new([(x, y - 5), (x + 16, y + 5)], GREEN_NEON.mix(0.5).filled())
            });

        // barras de previsão — com label para legenda
        chart
            .draw_series(preds.iter().enumerate().map(|(i, &p)| {
                let x = i as f64 + 0.6;
                Rectangle::new(
                    [(x + 0.27, 0.0), (x + 0.52, p.max(0.0))],
                    bar_color.mix(0.85).filled(),
                )
            }))
            .unwrap()
            .label("previsão")
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 16, y + 5)], bar_color.mix(0.85).filled())
            });

        // valor numérico acima de cada barra de previsão
        for (i, &p) in preds.iter().enumerate() {
            let x = i as f64 + 0.6 + 0.27;
            let y = p.max(0.0) + max_val * 0.02;
            chart
                .draw_series(std::iter::once(Text::new(
                    format!("{:.1}", p),
                    (x, y),
                    ("sans-serif", 11).into_font().color(&bar_color),
                )))
                .unwrap();
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .border_style(TEXT.mix(0.35))
            .background_style(BG.mix(0.9))
            .label_font(("sans-serif", 11).into_font().color(&TEXT))
            .draw()
            .unwrap();
    };

    draw_panel(&left, before, "Antes do treino", PURPLE);
    draw_panel(&right, after, "Depois do treino", GREEN_NEON);

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_activations(path: &str) {
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
