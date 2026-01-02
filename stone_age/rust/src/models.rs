use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationSettings {
    pub instrument_type: String,
    pub region: String,
    pub universe: String,
    pub delay: i32,
    pub decay: i32,
    pub neutralization: String,
    pub truncation: f64,
    pub pasteurization: String,
    pub unit_handling: String,
    pub nan_handling: String,
    pub language: String,
    pub visualization: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationRequest {
    pub r#type: String,
    pub settings: SimulationSettings,
    pub regular: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlphaResult {
    pub id: String,
    pub grade: String,
    pub is: AlphaMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlphaMetrics {
    pub fitness: f64,
    pub sharpe: f64,
    pub turnover: f64,
    pub returns: f64,
    pub checks: Vec<Check>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub result: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HopefulAlpha {
    pub expression: String,
    pub timestamp: i64,
    pub alpha_id: Option<String>,
    pub fitness: Option<f64>,
    pub sharpe: Option<f64>,
    pub turnover: Option<f64>,
    pub returns: Option<f64>,
    pub grade: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmittedAlphasResponse {
    pub count: i32,
    pub next: Option<String>,
    pub previous: Option<String>,
    pub results: Vec<SubmittedAlpha>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmittedAlpha {
    pub id: String,
    pub regular: RegularAlpha,
    pub grade: String,
    pub is: Option<AlphaMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegularAlpha {
    pub code: String,
    pub description: Option<String>,
    pub operator_count: i32,
} 