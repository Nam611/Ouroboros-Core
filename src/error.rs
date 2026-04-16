use thiserror::Error;

#[derive(Error, Debug)]
pub enum OuroborosError {
    #[error("Lỗi Hệ thống: {0}")]
    System(String),
    #[error("Lỗi Memory (TurboQuant): {0}")]
    Quantization(String),
    #[error("Lỗi Suy luận (LoopLM): {0}")]
    Reasoning(String),
    #[error("Lỗi Swarm: {0}")]
    Swarm(String),
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    
    // 🚨 BẢN VÁ: Đã bổ sung định dạng in lỗi ra màn hình
    #[error("Lỗi cấp phát RAM (Memory): {0}")]
    Memory(String),
}

pub type Result<T> = std::result::Result<T, OuroborosError>;