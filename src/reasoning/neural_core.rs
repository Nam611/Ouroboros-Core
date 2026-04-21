use crate::error::Result;
use bytes::Bytes;
use async_trait::async_trait;

/// Bản hợp đồng giao tiếp chuẩn giữa Trái tim (LoopLM) và Não bộ (NeuralEngine)
#[async_trait]
pub trait NeuralEngine: Send + Sync {
    /// Nạp trọng số vật lý từ đĩa vào RAM/VRAM
    async fn load_weights(&mut self, model_path: &str) -> Result<()>;
    
    /// 🚨 BẢN HỢP ĐỒNG KỶ NGUYÊN 3: Trả về (Token_ID, Chữ_Cái)
    /// Đẩy Tensor qua mạng và trả về duy nhất Token tiếp theo cùng đoạn text giải mã
    async fn forward_pass(&mut self, input_tensor: &Bytes, generated_tokens: &[u32]) -> Result<(u32, String)>;
    
    /// Quản lý vòng đời bộ nhớ, quét rác VRAM
    fn purge_vram(&self) -> Result<()>;
}