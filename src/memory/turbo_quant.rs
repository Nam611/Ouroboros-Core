use crate::error::{OuroborosError, Result};
use tracing::{debug, info};

/// 📦 [KHÔNG GIAN LƯỢNG TỬ]
/// Cấu trúc lưu trữ Vector đã bị ép xuống 8-bit (Tiết kiệm 75% RAM)
pub struct QuantizedCache {
    pub data: Vec<i8>,
    pub scale: f32, // Chìa khóa để giải nén lại thành f32
    pub original_len: usize,
}

pub struct TurboQuant {
    #[allow(dead_code)]
    compression_ratio: f32,
}

impl TurboQuant {
    pub fn new() -> Self {
        info!("⚡ [TurboQuant] Khởi tạo Lò phản ứng Lượng tử hóa Vector (f32 -> i8).");
        Self {
            compression_ratio: 4.0, // Từ 32-bit xuống 8-bit = Tỷ lệ 4x
        }
    }

    /// 🗜️ LÕI NÉN CỰC HẠN (SIMD-OPTIMIZED HOT-LOOP)
    /// Tối ưu hóa tận đáy L1 Cache. Không cấp phát động (Zero Re-allocation).
    pub fn compress_kv_cache(&self, raw_cache: &[f32]) -> Result<QuantizedCache> {
        let len = raw_cache.len();
        if len == 0 {
            return Err(OuroborosError::Memory("KV Cache rỗng, không thể lượng tử hóa.".into()));
        }

        // 1. TÌM CỰC ĐẠI BẰNG CHUNKING (Ép LLVM dùng lệnh SIMD MAX)
        let mut max_abs = 0.0f32;
        let mut chunks = raw_cache.chunks_exact(8); // Ép CPU nạp 8 float (256-bit) vào thanh ghi AVX
        
        for chunk in &mut chunks {
            // Unroll bằng tay để không có nhánh rẽ (No Branching)
            let m0 = chunk[0].abs().max(chunk[1].abs());
            let m1 = chunk[2].abs().max(chunk[3].abs());
            let m2 = chunk[4].abs().max(chunk[5].abs());
            let m3 = chunk[6].abs().max(chunk[7].abs());
            let m_half1 = m0.max(m1);
            let m_half2 = m2.max(m3);
            max_abs = max_abs.max(m_half1.max(m_half2));
        }
        
        // Quét phần dư (remainder) không chia hết cho 8
        for &val in chunks.remainder() {
            max_abs = max_abs.max(val.abs());
        }

        // 2. TOÁN HỌC NGHỊCH ĐẢO (FMA - Fused Multiply-Add Preparation)
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv_scale = 1.0 / scale;

        // 3. ZERO-ALLOCATION QUANTIZATION (Cấp phát RAM 1 lần duy nhất)
        // Báo trước cho Hệ điều hành chính xác số Byte cần dùng để tránh Re-allocation
        let mut quantized_data: Vec<i8> = Vec::with_capacity(len);
        let ptr = quantized_data.as_mut_ptr();

        // Ép xung tiến trình ghi trực tiếp vào RAM (Bypass bounds checking an toàn)
        unsafe {
            for i in 0..len {
                // Kỹ thuật nhân cực tốc (FMA) thay vì chia
                let scaled = *raw_cache.get_unchecked(i) * inv_scale;
                
                // Toán học thao tác Bit (Bitwise) để thay thế .clamp() tốn kém
                // Ép kiểu float -> int nhanh nhất trên kiến trúc x86_64
                let mut q = scaled.round() as i32;
                
                // Branchless Clamp (Ép giới hạn không dùng if/else)
                q = q.min(127).max(-127);
                
                // Ghi thẳng vào địa chỉ RAM
                ptr.add(i).write(q as i8);
            }
            // Xác nhận với bộ đếm RAM của Rust rằng ta đã ghi xong
            quantized_data.set_len(len);
        }

        debug!("🔮 [TurboQuant-SIMD] Nén {} Floats thành công. Scale: {}", len, scale);

        Ok(QuantizedCache {
            data: quantized_data,
            scale,
            original_len: len,
        })
    }

    /// 🔓 LÕI GIẢI NÉN (DECOMPRESSION)
    /// Phục hồi i8 về f32 khi Mạng Nơ-ron cần đọc lại Ký ức
    #[allow(dead_code)]
    pub fn decompress_kv_cache(&self, q_cache: &QuantizedCache) -> Result<Vec<f32>> {
        let scale = q_cache.scale;
        
        // Tái tạo lại ma trận xấp xỉ
        let restored_data: Vec<f32> = q_cache.data.iter()
            .map(|&q| (q as f32) * scale)
            .collect();

        Ok(restored_data)
    }
}