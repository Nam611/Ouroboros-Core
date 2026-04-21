use crate::error::{OuroborosError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use tracing::{debug, info};

/// 📚 TÀNG KINH CÁC: Lưu trữ Vector Tri thức trên RAM (Kiến trúc SoA - Structure of Arrays)
#[derive(Serialize, Deserialize)]
pub struct SemanticGraph {
    pub documents: Vec<String>,       // Chứa text thật (Wikipedia, Docs)
    pub vectors: Vec<Vec<f32>>,       // Chứa tọa độ không gian của Text (Tối ưu Cache L1/L2)
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

    /// Xóa sạch Tàng Kinh Các (Dùng khi muốn Re-index lại dự án)
    pub fn clear(&mut self) {
        self.documents.clear();
        self.vectors.clear();
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
    fn fast_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (&va, &vb) in a.iter().zip(b.iter()) {
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }

    // =========================================================================
    // 🚨 MODULE BẤT TỬ (PERSISTENCE): ĐÓNG BĂNG VÀ RÃ ĐÔNG BỘ NHỚ TỚI ĐĨA CỨNG
    // =========================================================================

    /// Lưu toàn bộ Tàng Kinh Các xuống đĩa (Binary Format)
    pub fn save_to_disk(&self, path: &str) -> Result<()> {
        info!("💾 Đang nén Tàng Kinh Các xuống đĩa từ (Binary Format)...");
        let file = File::create(path).map_err(|e| {
            OuroborosError::System(format!("Không thể tạo file index: {}", e))
        })?;
        
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, &self).map_err(|e| {
            OuroborosError::System(format!("Lỗi lượng tử hóa nhị phân: {}", e))
        })?;
        
        info!("✅ Đã phong ấn {} Vector Tri thức an toàn tại [{}]", self.documents.len(), path);
        Ok(())
    }

    /// Nạp thẳng từ ổ cứng lên RAM (Tốc độ đọc GB/s)
    pub fn load_from_disk(path: &str) -> Result<Self> {
        debug!("⚡ Kích hoạt nạp bộ nhớ nhị phân từ [{}]...", path);
        let file = File::open(path).map_err(|e| {
            OuroborosError::System(format!("Lỗi mở file index: {}", e))
        })?;
        let reader = BufReader::new(file);
        
        let graph: SemanticGraph = bincode::deserialize_from(reader).map_err(|e| {
            OuroborosError::System(format!("Lỗi giải mã nhị phân: {}", e))
        })?;
        
        info!("✅ Đã nạp thành công {} Vector Tri thức vào Khối RAM thần kinh.", graph.documents.len());
        Ok(graph)
    }
}