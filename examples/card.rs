use fcsrv::{model::ModelType, BootArgs};
use ort::AllocatorType;
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let args = BootArgs {
        debug: false,
        bind: "0.0.0.0:8000".parse().unwrap(),
        tls_cert: None,
        tls_key: None,
        api_key: None,
        multi_image_limit: 1,
        update_check: false,
        model_dir: Some(PathBuf::from("models")),
        num_threads: 4,
        allocator: AllocatorType::Arena,
        fallback_solver: None,
        fallback_key: None,
        fallback_image_limit: 3,
        fallback_endpoint: None,
    };

    let predictor = fcsrv::model::get_predictor(ModelType::Card, &args)
        .await
        .unwrap();

    let image_file = std::fs::read("images/card/card_1.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 0);

    let image_file = std::fs::read("images/card/card_2.jpg").unwrap();
    let guess = predictor
        .predict(image::load_from_memory(&image_file).unwrap())
        .unwrap();
    assert_eq!(guess, 2);
}
