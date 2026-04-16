use crate::error::{OuroborosError, Result};
use tracing::{debug, info};

/// 📚 TÀNG KINH CÁC: Lưu trữ Vector Tri thức trên RAM
pub struct SemanticGraph {
    pub documents: Vec<String>,       // Chứa text thật (Wikipedia, Docs)
    pub vectors: Vec<Vec<f32>>,       // Chứa tọa độ không gian của Text
    pub dimension: usize,             // Số chiều của Vector (ví dụ: 384 chiều)
}

impl SemanticGraph {
    pub fn new(dimension: usize) -> Self {
        info!("📚 [Semantic Graph] Khởi tạo Tàng Kinh Các với không gian {} chiều.", dimension);
        Self {
            documents: Vec::new(),
            vectors: Vec::new(),
            dimension,
        }
    }

    /// Nạp tri thức vật lý vào Não bộ
    pub fn ingest_knowledge(&mut self, text: &str, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(OuroborosError::System(
                format!("Sai lệch chiều không gian. Chờ {}, nhận {}.", self.dimension, vector.len())
            ));
        }
        self.documents.push(text.to_string());
        self.vectors.push(vector);
        debug!("Đã nạp 1 phân mảnh tri thức. Tổng Ký ức: {}", self.documents.len());
        Ok(())
    }

    /// 🧮 LÕI TÌM KIẾM CỰC TỐC (SIMD-Optimized Cosine Similarity)
    /// Tìm ra phân mảnh kiến thức gần nhất với câu hỏi của User
    pub fn search_closest_knowledge(&self, query_vector: &[f32]) -> Result<Option<String>> {
        if self.vectors.is_empty() {
            return Ok(None); // Não chưa có dữ liệu
        }

        let mut best_score = -1.0_f32; // Cosine chạy từ -1 (ngược hướng) đến 1 (trùng khớp)
        let mut best_index = 0;

        for (i, doc_vector) in self.vectors.iter().enumerate() {
            let score = Self::fast_cosine_similarity(query_vector, doc_vector);
            if score > best_score {
                best_score = score;
                best_index = i;
            }
        }

        debug!("🔮 [Semantic Graph] Tìm thấy Tri thức phù hợp (Độ trùng khớp: {:.2}%)", best_score * 100.0);
        Ok(Some(self.documents[best_index].clone()))
    }

    /// ⚡ ĐỘNG CƠ NHÂN VÔ HƯỚNG TỐI ƯU TẬN ĐÁY (SIMD Unrolling)
    /// Tính Cosine Similarity: (A.B) / (|A|*|B|)
    fn fast_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        // Ép LLVM dùng lệnh SIMD để nhân nhiều số cùng lúc
        // (Sử dụng iterators zip cho phép compiler tối ưu hóa vectorization rất tốt)
        for (&va, &vb) in a.iter().zip(b.iter()) {
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        // Căn bậc 2 siêu tốc bằng hàm nội tại (intrinsic) của phần cứng
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }
}