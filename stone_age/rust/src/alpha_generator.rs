use crate::{api_client::ApiClient, models::*};
use anyhow::Result;
use log::{info, warn, error};
use regex::Regex;
use std::{collections::HashMap, time::{Duration, SystemTime, UNIX_EPOCH}};
use tokio::time::sleep;
use futures::future::join_all;

pub struct AlphaGenerator {
    api_client: ApiClient,
    pending_simulations: HashMap<String, String>, // url -> expression
    rate_limiter: RateLimiter,
}

struct RateLimiter {
    max_requests: usize,
    time_window: Duration,
    request_times: Vec<SystemTime>,
}

impl RateLimiter {
    fn new(max_requests: usize, time_window: Duration) -> Self {
        Self {
            max_requests,
            time_window,
            request_times: Vec::new(),
        }
    }

    async fn wait_for_slot(&mut self) {
        let now = SystemTime::now();
        self.request_times.retain(|&time| {
            time.duration_since(now).unwrap_or(Duration::from_secs(0)) < self.time_window
        });

        while self.request_times.len() >= self.max_requests {
            sleep(Duration::from_secs(1)).await;
            self.request_times.retain(|&time| {
                time.duration_since(now).unwrap_or(Duration::from_secs(0)) < self.time_window
            });
        }

        self.request_times.push(now);
    }
}

impl AlphaGenerator {
    pub fn new(username: &str, password: &str) -> Result<Self> {
        Ok(Self {
            api_client: ApiClient::new(username, password)?,
            pending_simulations: HashMap::new(),
            rate_limiter: RateLimiter::new(8, Duration::from_secs(60)),
        })
    }

    pub async fn test_alpha_batch(&mut self, mut expressions: Vec<String>) -> Result<Vec<AlphaResult>> {
        // Get existing expressions first
        let existing = self.get_existing_expressions().await?;
        
        // Filter out similar expressions
        expressions.retain(|expr| !self.is_similar_to_existing(expr, &existing));
        
        if expressions.is_empty() {
            info!("All expressions were too similar to existing ones");
            return Ok(Vec::new());
        }

        info!("Testing {} unique expressions after filtering", expressions.len());
        
        // Submit all simulations
        for expression in expressions {
            self.rate_limiter.wait_for_slot().await;
            match self.api_client.submit_simulation(&expression).await {
                Ok(url) => {
                    info!("Successfully submitted: {}", expression);
                    self.pending_simulations.insert(url, expression);
                }
                Err(e) => {
                    error!("Failed to submit {}: {}", expression, e);
                }
            }
        }

        // Monitor pending simulations
        while !self.pending_simulations.is_empty() {
            info!("Monitoring {} pending simulations...", self.pending_simulations.len());
            let mut completed = Vec::new();
            
            for (url, expression) in &self.pending_simulations {
                match self.api_client.check_simulation(url).await {
                    Ok(Some(result)) => {
                        info!("Simulation complete for {}", expression);
                        completed.push(url.clone());
                    }
                    Ok(None) => {
                        // Need to wait more
                        continue;
                    }
                    Err(e) => {
                        error!("Error checking simulation {}: {}", url, e);
                        completed.push(url.clone());
                    }
                }
            }

            // Remove completed simulations
            for url in completed {
                self.pending_simulations.remove(&url);
            }

            if !self.pending_simulations.is_empty() {
                sleep(Duration::from_secs(5)).await;
            }
        }

        Ok(Vec::new())
    }

    pub fn generate_parameter_variations(&self, expression: &str) -> Result<Vec<String>> {
        let re = Regex::new(r"\d+")?;
        let mut variations = Vec::new();
        let mut positions = Vec::new();
        let mut base_values = Vec::new();

        // Find all numeric parameters
        for cap in re.find_iter(expression) {
            let value: i32 = cap.as_str().parse()?;
            positions.push((cap.start(), cap.end()));
            base_values.push(value);
        }

        if positions.is_empty() {
            return Ok(vec![expression.to_string()]);
        }

        // Generate variations for each parameter
        let mut param_variations = Vec::new();
        for &base_value in &base_values {
            let values = if base_value <= 10 {
                // For small values, test more granularly
                vec![
                    (base_value as f64 * 0.5).max(1.0) as i32,
                    (base_value as f64 * 0.75).max(1.0) as i32,
                    base_value,
                    (base_value as f64 * 1.25) as i32,
                    (base_value as f64 * 1.5) as i32,
                ]
            } else {
                // For larger values, test wider range
                let lower = (base_value as f64 * 0.5).max(1.0) as i32;
                let upper = (base_value as f64 * 1.5) as i32;
                let step = ((upper - lower) as f64 / 4.0).max(1.0) as i32;
                
                let mut values = Vec::new();
                let mut current = lower;
                while current <= upper {
                    values.push(current);
                    current += step;
                }
                if !values.contains(&base_value) {
                    values.push(base_value);
                    values.sort();
                }
                values
            };
            param_variations.push(values);
        }

        // Generate all combinations
        let mut stack = vec![(String::from(expression), 0, Vec::new())];
        while let Some((current, depth, params)) = stack.pop() {
            if depth == positions.len() {
                variations.push(current);
                continue;
            }

            for &value in &param_variations[depth] {
                let mut new_params = params.clone();
                new_params.push(value);
                
                let mut new_expr = current.clone();
                let offset = positions[depth].0;
                let len = positions[depth].1 - positions[depth].0;
                new_expr.replace_range(offset..offset+len, &value.to_string());
                
                stack.push((new_expr, depth + 1, new_params));
            }
        }

        Ok(variations)
    }

    pub async fn get_existing_expressions(&self) -> Result<Vec<String>> {
        let submitted = self.api_client.get_submitted_alphas().await?;
        Ok(submitted.results
            .into_iter()
            .map(|alpha| alpha.regular.code)
            .collect())
    }

    pub fn is_similar_to_existing(&self, new_expr: &str, existing: &[String]) -> bool {
        for expr in existing {
            // Check for exact matches
            if expr == new_expr {
                return true;
            }

            // Check for similar structure (same operators in similar order)
            let new_ops: Vec<_> = new_expr.split(|c: char| !c.is_alphabetic()).collect();
            let existing_ops: Vec<_> = expr.split(|c: char| !c.is_alphabetic()).collect();
            if new_ops == existing_ops {
                return true;
            }

            // Check for similar ratios/combinations
            let new_parts: Vec<_> = new_expr.split(&['(', ')', '/', '+', '-', '*', ','][..]).collect();
            let existing_parts: Vec<_> = expr.split(&['(', ')', '/', '+', '-', '*', ','][..]).collect();
            if new_parts.len() == existing_parts.len() {
                let matching = new_parts.iter()
                    .zip(existing_parts.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                if matching as f64 / new_parts.len() as f64 > 0.7 {
                    return true;
                }
            }
        }
        false
    }
} 