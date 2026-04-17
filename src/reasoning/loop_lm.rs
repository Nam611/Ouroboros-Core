use crate::error::{OuroborosError, Result};
use crate::reasoning::candle_bridge::CandleEngine;
use crate::reasoning::neural_core::NeuralEngine;
use bytes::Bytes;
use tracing::{debug, info};

pub struct LoopLM {
    engine: CandleEngine,
    max_loops: usize,
}

impl LoopLM {
    pub fn new() -> Self {
        Self {
            engine: CandleEngine::new().expect("❌ Lỗi khởi tạo Candle Bridge"),
            max_loops: 200, // Tăng nhẹ max_loops để chứa đủ JSON output
        }
    }

    /// 🧠 LÕI SUY LUẬN AUTO-REGRESSIVE (JSON FORCED)
    pub async fn process_reasoning(&mut self, raw_task: &str) -> Result<String> {
        // 🚨 KỶ LUẬT FIRST PRINCIPLES: ÉP BUỘC ĐỊNH DẠNG JSON
        let system_prompt = format!(
            "Bạn là Ouroboros Core, một Hệ điều hành AI siêu việt. \
            Nhiệm vụ của bạn là tối ưu hóa mã nguồn. BẠN CHỈ ĐƯỢC PHÉP TRẢ VỀ ĐÚNG MỘT KHỐI JSON DUY NHẤT. \
            Tuyệt đối không giải thích thêm. Định dạng bắt buộc: \
            {{\"target_file\": \"...\", \"original_function_signature\": \"...\", \"optimized_code\": \"...\", \"reasoning\": \"...\"}}\n\n\
            Nhiệm vụ hiện tại: {}\n\nJSON Output:", 
            raw_task
        );

        let mut current_input = system_prompt;
        let mut full_output = String::new();
        let mut loop_count = 0;

        loop {
            if loop_count >= self.max_loops {
                debug!("🚧 [SURVIVAL LIMIT] Chạm ngưỡng an toàn {} loops. Cắt luồng!", self.max_loops);
                break;
            }

            let input_bytes = Bytes::from(current_input.clone());
            let (token_id, output_text) = self.engine.forward_pass(&input_bytes).await?;

            full_output.push_str(&output_text);
            
            // In ra terminal thời gian thực
            print!("{}", output_text);
            use std::io::Write;
            let _ = std::io::stdout().flush();

            // 151643 là Token <|endoftext|> của Qwen2
            if token_id == 151643 || output_text.contains('}') {
                println!("\n🚪 [EXIT GATE] Khối JSON đã đóng lại ở vòng {}. Đóng buồng đốt!", loop_count);
                break;
            }

            current_input = output_text;
            loop_count += 1;
        }

        // 🚨 CỔNG KIỂM DUYỆT JSON TẠI CHỖ (Runtime Validation)
        self.validate_json_contract(&full_output)?;

        Ok(full_output)
    }

    /// ⚖️ TITANIUM GATE: Kiểm tra JSON có hợp lệ không trước khi chuyển cho Agent
    fn validate_json_contract(&self, raw_output: &str) -> Result<()> {
        // Trích xuất phần nằm trong dấu ngoặc nhọn {...} đề phòng model nói lảm nhảm
        let json_start = raw_output.find('{');
        let json_end = raw_output.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let clean_json = &raw_output[start..=end];
            
            // Thử parse xem có đúng cấu trúc FileMutationContract không
            match serde_json::from_str::<crate::swarm::tools::FileMutationContract>(clean_json) {
                Ok(_) => {
                    info!("✅ [Data Contract] Não bộ sinh ra Hợp đồng JSON hoàn hảo!");
                    Ok(())
                },
                Err(e) => Err(OuroborosError::Reasoning(format!("Cấu trúc JSON bị vỡ: {}", e)))
            }
        } else {
            Err(OuroborosError::Reasoning("Model không sinh ra khối JSON nào!".into()))
        }
    }
}