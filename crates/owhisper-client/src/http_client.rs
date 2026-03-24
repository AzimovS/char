use std::time::Duration;

use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_tracing::TracingMiddleware;

pub fn create_client() -> ClientWithMiddleware {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("failed to build reqwest client");

    ClientBuilder::new(client)
        .with(TracingMiddleware::default())
        .build()
}

pub fn timeout_for_file_size(file_size_bytes: u64) -> Duration {
    let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);
    let secs = (file_size_mb * 30.0).max(300.0) as u64;
    Duration::from_secs(secs)
}
