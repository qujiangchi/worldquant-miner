use crate::models::*;
use anyhow::{Context, Result};
use reqwest::{Client, StatusCode};
use std::time::Duration;

pub struct ApiClient {
    client: Client,
    base_url: String,
    auth_token: String,
}

impl ApiClient {
    pub fn new(username: &str, password: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
            
        let mut api_client = Self {
            client,
            base_url: "https://api.worldquantbrain.com".to_string(),
            auth_token: String::new(),
        };
        
        api_client.authenticate(username, password)?;
        Ok(api_client)
    }
    
    async fn authenticate(&mut self, username: &str, password: &str) -> Result<()> {
        let response = self.client
            .post(format!("{}/authentication", self.base_url))
            .basic_auth(username, Some(password))
            .send()
            .await?;
            
        if response.status() != StatusCode::CREATED {
            anyhow::bail!("Authentication failed: {}", response.text().await?);
        }
        
        self.auth_token = response
            .headers()
            .get("authorization")
            .context("No auth token in response")?
            .to_str()?
            .to_string();
            
        Ok(())
    }
    
    pub async fn submit_simulation(&self, expression: &str) -> Result<String> {
        let request = SimulationRequest {
            r#type: "REGULAR".to_string(),
            settings: SimulationSettings {
                instrument_type: "EQUITY".to_string(),
                region: "USA".to_string(),
                universe: "TOP3000".to_string(),
                delay: 1,
                decay: 0,
                neutralization: "INDUSTRY".to_string(),
                truncation: 0.08,
                pasteurization: "ON".to_string(),
                unit_handling: "VERIFY".to_string(),
                nan_handling: "OFF".to_string(),
                language: "FASTEXPR".to_string(),
                visualization: false,
            },
            regular: expression.to_string(),
        };
        
        let response = self.client
            .post(format!("{}/simulations", self.base_url))
            .bearer_auth(&self.auth_token)
            .json(&request)
            .send()
            .await?;
            
        if response.status() != StatusCode::CREATED {
            anyhow::bail!("Simulation submission failed: {}", response.text().await?);
        }
        
        let location = response
            .headers()
            .get("location")
            .context("No progress URL in response")?
            .to_str()?
            .to_string();
            
        Ok(location)
    }
    
    pub async fn check_simulation(&self, progress_url: &str) -> Result<Option<AlphaResult>> {
        let response = self.client
            .get(progress_url)
            .bearer_auth(&self.auth_token)
            .send()
            .await?;
            
        if let Some(retry_after) = response.headers().get("retry-after") {
            let seconds: u64 = retry_after.to_str()?.parse()?;
            tokio::time::sleep(Duration::from_secs(seconds)).await;
            return Ok(None);
        }
        
        let result: AlphaResult = response.json().await?;
        Ok(Some(result))
    }

    pub async fn get_submitted_alphas(&self) -> Result<SubmittedAlphasResponse> {
        let url = format!(
            "{}/users/self/alphas?limit=50&offset=0&status!=UNSUBMITTED%1FIS-FAIL&order=-dateCreated&hidden=false",
            self.base_url
        );
        
        let response = self.client
            .get(&url)
            .bearer_auth(&self.auth_token)
            .send()
            .await?;
            
        if response.status() != StatusCode::OK {
            anyhow::bail!("Failed to fetch submitted alphas: {}", response.text().await?);
        }
        
        let alphas: SubmittedAlphasResponse = response.json().await?;
        Ok(alphas)
    }
} 