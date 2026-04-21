use crate::error::Result;
use crate::reasoning::candle_bridge::CandleEngine;
use bytes::Bytes;
use tracing::debug;

pub struct LoopLM {
    engine: CandleEngine,
    max_loops: usize,
}

impl LoopLM {
    pub fn new() -> Self {
        Self {
            engine: CandleEngine::new().expect("❌ Lỗi khởi tạo Candle Bridge"),
            max_loops: 200, 
        }
    }

    /// 🧠 LÕI SUY LUẬN MÃ NGUỒN PURE (PURE CODE GENERATION)
    pub async fn process_reasoning(&mut self, raw_task: &str) -> Result<String> {
        // 🚨 KỶ LUẬT FIRST PRINCIPLES: CUNG CẤP VÍ DỤ KHÓA CHẶT ĐỊNH DẠNG (FEW-SHOT)
        let system_prompt = format!(
            "Bạn là Ouroboros Core, một Kỹ sư phần mềm Rust. \
            Nhiệm vụ: Viết mã nguồn theo yêu cầu. BẠN CHỈ ĐƯỢC PHÉP VIẾT CODE RUST. KHÔNG GIẢI THÍCH.\n\n\
            VÍ DỤ CHUẨN:\n\
            ```rust\n\
            fn main() {{\n    println!(\"System Online\");\n}}\n\
            ```\n\n\
            NHIỆM VỤ: {}\n\nCODE RUST OUTPUT:\n",
            raw_task
        );

        let mut current_input = system_prompt;
        let mut full_output = String::new();
        let mut loop_count = 0;
        let mut token_history: Vec<u32> = Vec::with_capacity(self.max_loops);

        loop {
            if loop_count >= self.max_loops {
                debug!("🚧 [SURVIVAL LIMIT] Cắt luồng ở vòng {}.", self.max_loops);
                break;
            }

            let input_bytes = Bytes::from(current_input.clone());
            let (token_id, output_text) = self.engine.forward_pass(&input_bytes, &token_history).await?;

            token_history.push(token_id);
            full_output.push_str(&output_text);
            
            print!("{}", output_text);
            use std::io::Write;
            let _ = std::io::stdout().flush();

            // Kết thúc khi gặp token đóng
            if token_id == 151643 {
                println!("\n🚪 [EXIT GATE] Suy luận hoàn tất ở vòng {}.", loop_count);
                break;
            }

            current_input = output_text;
            loop_count += 1;
        }

        // Không cần validate JSON ở đây nữa
        Ok(full_output)
    }
}