use tch::Tensor;

pub trait ToTensor {
    fn to_tensor(&self) -> Tensor;
}

impl ToTensor for [f32; 4] {
    fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(self).unsqueeze(0)
    }
}

pub fn plot_rewards(rewards: &Vec<f32>, filename: &str, title: &str) {
    use plotters::prelude::*;

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_reward = rewards.iter().cloned().fold(f32::MIN, f32::max).max(1.0); // 避免空图或全为 0

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("Times New Roman", 32).into_font())
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..rewards.len(), 0f32..max_reward)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Episode")
        .y_desc("Total Reward")
        .axis_desc_style(("Times New Roman", 22))
        .label_style(("Times New Roman", 18))
        .light_line_style(&WHITE.mix(0.3))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            rewards.iter().enumerate().map(|(i, r)| (i, *r)),
            &BLUE,
        ))
        .unwrap()
        .label("Reward")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("Times New Roman", 18))
        .draw()
        .unwrap();

    println!("Saved training plot to {}", filename);
}
