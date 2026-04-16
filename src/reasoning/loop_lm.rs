use crate::error::Result;
use crate::reasoning::neural_core::NeuralEngine;
use tracing::{debug, info, warn};
use std::io::{Write, stdout};

pub struct LoopLM {
    pub neural_backend: Option<Box<dyn NeuralEngine>>,
    max_loops: usize,
}

impl LoopLM {
    pub fn new() -> Self {
        info!("⚙️ Khởi tạo LoopLM (Trái tim Ouroboros)...");
        
        // Cắm Não bộ vật lý vào hệ thống
        let backend: Option<Box<dyn NeuralEngine>> = 
            match crate::reasoning::candle_bridge::CandleEngine::new() {
                Ok(engine) => Some(Box::new(engine)),
                Err(e) => {
                    warn!("⚠️ Không thể khởi tạo Lõi Tensor: {}. Ouroboros sẽ chạy mù.", e);
                    None
                }
            };

        Self {
            neural_backend: backend,
            // 🚨 GIỚI HẠN SINH TỒN: Cho phép nói tối đa 64 từ mỗi chu kỳ để bảo vệ RAM
            max_loops: 64, 
        }
    }

    pub async fn process_reasoning(&mut self, task: &str) -> Result<String> {
        info!("🧠 [Ouroboros Loop] Kích hoạt suy luận Tự hồi quy (Auto-Regressive). Task: [{}]", task);

        let mut current_context = task.to_string();
        let mut final_response = String::new();

        // 🚨 KÍCH HOẠT HIỆU ỨNG TRUYỀN DỮ LIỆU THỜI GIAN THỰC (STREAMING)
        print!("\n================== [ OUROBOROS SIGNAL ] ==================\n");
        print!("💬 Ouroboros: ");
        let _ = stdout().flush(); // Ép hệ điều hành in ngay lập tức ra màn hình

        if let Some(engine) = &self.neural_backend {
            for loop_idx in 1..=self.max_loops {
                let bytes_input = bytes::Bytes::from(current_context.clone());
                
                // Trái tim gọi Não bộ: Nhận về Token ID và Chữ cái dịch được
                let (token_id, new_word) = engine.forward_pass(&bytes_input).await?;

                // 🚨 GIAO THỨC EXIT GATE (Tuyệt đối quan trọng)
                // Token ID 151643 (<|endoftext|>) hoặc 151645 (<|im_end|>) là dấu chấm hết của Qwen.
                // Khi mô hình nhả mã này, nghĩa là nó đã trả lời xong.
                if token_id == 151643 || token_id == 151645 || new_word.is_empty() {
                    debug!("\n🚪 [EXIT GATE] Tư duy đã hội tụ ở vòng {}. Đóng buồng đốt!", loop_idx);
                    break;
                }

                // In chữ cái vừa nghĩ ra thẳng lên Terminal (Không đợi hết câu)
                print!("{}", new_word);
                let _ = stdout().flush();

                // Nối từ mới vào đuôi để làm ngữ cảnh cho vòng lặp tương lai
                current_context.push_str(&new_word);
                final_response.push_str(&new_word);
            }
            print!("\n==========================================================\n");
        } else {
            warn!("❌ Không có lõi vật lý. Ouroboros không thể phát âm.");
            final_response = "ERROR: NO_PHYSICAL_CORE".to_string();
        }

        Ok(final_response)
    }
}