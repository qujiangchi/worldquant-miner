mod api_client;
mod models;
mod alpha_generator;

use alpha_generator::AlphaGenerator;
use clap::Parser;
use anyhow::Result;
use log::{info, error};
use std::{path::PathBuf, time::Duration};
use tokio::time::sleep;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "./credential.txt")]
    credentials: PathBuf,
    
    #[arg(short, long, default_value = "./results")]
    output_dir: PathBuf,
    
    #[arg(short, long, default_value_t = 5)]
    batch_size: usize,
    
    #[arg(short, long, default_value_t = 60)]
    sleep_time: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Read credentials
    let creds: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(&args.credentials)?
    )?;
    
    if creds.len() != 2 {
        anyhow::bail!("Credentials file should contain [username, password]");
    }

    let mut generator = AlphaGenerator::new(&creds[0], &creds[1])?;
    
    loop {
        match std::fs::read_to_string("hopeful_alphas.json") {
            Ok(content) => {
                let hopeful_alphas: Vec<models::HopefulAlpha> = serde_json::from_str(&content)?;
                if hopeful_alphas.is_empty() {
                    info!("No hopeful alphas to process. Waiting...");
                    sleep(Duration::from_secs(args.sleep_time)).await;
                    continue;
                }

                for alpha in hopeful_alphas {
                    info!("Processing alpha: {}", alpha.expression);
                    let variations = generator.generate_parameter_variations(&alpha.expression)?;
                    let results = generator.test_alpha_batch(variations).await?;
                    
                    // Save results
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_secs();
                    
                    let output_file = args.output_dir.join(format!("results_{}.json", timestamp));
                    std::fs::write(
                        output_file,
                        serde_json::to_string_pretty(&results)?
                    )?;
                }
            }
            Err(e) => {
                error!("Error reading hopeful_alphas.json: {}", e);
                sleep(Duration::from_secs(300)).await;
            }
        }
        
        sleep(Duration::from_secs(args.sleep_time)).await;
    }
} 