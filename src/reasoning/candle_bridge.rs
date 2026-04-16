use crate::error::{OuroborosError, Result};
use crate::reasoning::neural_core::NeuralEngine;
use async_trait::async_trait;
use bytes::Bytes;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel};
// 🚨 LÕI VẬT LÝ THỐNG KÊ (Statistical Mechanics Core)
use candle_transformers::generation::LogitsProcessor; 
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Lõi Xử lý Tensor thuần Rust (Tích hợp Qwen2.5-0.5B Native)
pub struct CandleEngine {
    device: Device,
    model: Mutex<QwenModel>,
    tokenizer: Tokenizer,
    seqlen_offset: Mutex<usize>,
    // 🚨 BỌC MUTEX: Khởi tạo 1 lần, dùng vĩnh viễn, không xả rác RAM
    logits_processor: Mutex<LogitsProcessor>, 
}

impl CandleEngine {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("⚙️ [Candle Bridge] Kích hoạt buồng đốt Tensor trên thiết bị: {:?}", device);
        
        info!("🔥 [TITANIUM PROTOCOL] Gỡ bỏ hf-hub. Kích hoạt Native OS Downloader...");
        let (config_path, tokenizer_path, weights_path) = Self::download_core_files()?;

        info!("🧩 Cấp phát Tokenizer siêu tốc...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi phân tích Tokenizer: {}", e)))?;

        info!("🧠 Đúc Mạch Nơ-ron (Qwen2 Architecture) vào VRAM/RAM...");
        let config: QwenConfig = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|e| OuroborosError::Reasoning(e.to_string()))?
        ).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[&weights_path], candle_core::DType::F32, &device)
                .map_err(|e| OuroborosError::Reasoning(format!("Lỗi Memory-Map Tensor: {}", e)))? 
        };
        
        let model = QwenModel::new(&config, vb)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi ráp mạch Transformer: {}", e)))?;

        // 🚨 THIẾT LẬP THÔNG SỐ TOÁN HỌC (The Golden Ratio)
        // Seed: 299792458 (Tốc độ ánh sáng - Đảm bảo tính tất định nhưng vẫn hỗn mang)
        // Temperature: 0.7 (Nung nóng ma trận)
        // Top-P: 0.95 (Chặt đứt đuôi rác)
        let logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.95));

        info!("✅ [Neural Boot] Ouroboros đã mở mắt. Cấy ghép Dây thanh quản Lượng tử.");

        Ok(Self { 
            device, 
            model: Mutex::new(model), 
            tokenizer,
            seqlen_offset: Mutex::new(0),
            logits_processor: Mutex::new(logits_processor), // Cố định vào RAM
        })
    }

    /// Động cơ I/O cấp thấp: Gọi thẳng cURL của Linux, vô nhiễm với mọi bug của Rust
    fn download_core_files() -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        let cache_dir = std::path::PathBuf::from(".ouroboros_cache");
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| OuroborosError::Reasoning(format!("Lỗi tạo thư mục: {}", e)))?;
        }

        let base_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main";
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        let mut paths = vec![];

        for file in files {
            let file_path = cache_dir.join(file);
            paths.push(file_path.clone());
            
            if file_path.exists() {
                info!("✅ Đã có sẵn lõi tại Local: {}", file);
                continue;
            }
            
            info!("⬇️ Giao thức cURL: Đang xuyên thủng mạng để kéo [{}]...", file);
            let url = format!("{}/{}", base_url, file);
            
            // Gọi syscall xuống nhân Linux: curl -L -# -o <đích> <url>
            let status = std::process::Command::new("curl")
                .args(["-L", "-#", "-o", file_path.to_str().unwrap(), &url])
                .env("NO_PROXY", "*") // Thiết quân luật tuyệt đối
                .env("no_proxy", "*")
                .status()
                .map_err(|e| OuroborosError::Reasoning(format!("OS từ chối lệnh cURL: {}", e)))?;
                
            if !status.success() {
                let _ = std::fs::remove_file(&file_path); // Dọn rác nếu tải xịt
                return Err(OuroborosError::Reasoning(format!("cURL thất bại khi tải {}", file)));
            }
        }
        
        Ok((paths[0].clone(), paths[1].clone(), paths[2].clone()))
    }
}

#[async_trait]
impl NeuralEngine for CandleEngine {
    async fn load_weights(&mut self, _model_path: &str) -> Result<()> { 
        Ok(()) 
    }

    async fn forward_pass(&self, input_tensor: &Bytes) -> Result<(u32, String)> {
        let text = String::from_utf8_lossy(input_tensor).to_string();
        debug!("⚡ Não bộ tiếp nhận kích thích: [{}]", text);

        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let token_ids = tokens.get_ids();

        let mut offset = self.seqlen_offset.lock().unwrap();
        let mut model = self.model.lock().unwrap();

        // Thuật toán Tensor Time-Keeper
        if token_ids.len() <= *offset {
            debug!("🔄 Kích hoạt Zero-State. Đặt lại Không-Thời gian Tensor.");
            model.clear_kv_cache();
            *offset = 0;
        }

        // Trích xuất Token mới nhất
        let new_tokens = &token_ids[*offset..];
        let input_t = Tensor::new(new_tokens, &self.device)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?.unsqueeze(0)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        debug!("🧮 Đẩy {} Token mới qua Transformer tại vị trí Offset: {}...", new_tokens.len(), *offset);

        // Chạy qua Transformer
        let logits = model.forward(&input_t, *offset)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi tính toán lõi: {}", e)))?;

        *offset += new_tokens.len();

        let sq_logits = logits.squeeze(0).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        // ... (Logic offset và tính logits giữ nguyên)
        let actual_dim = sq_logits.dim(0).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        
        let last_token_logits = sq_logits.get(actual_dim - 1)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        // 🚨 KỶ NGUYÊN 3.1: TOÁN HỌC SAMPLING TỐI ƯU (O(1) Memory Allocation)
        // Lấy con trỏ khóa Mutex
        let mut lp = self.logits_processor.lock().unwrap();
        
        // Sampling trực tiếp từ Logits thô (Bỏ qua bước trung gian Softmax tốn CPU)
        let token_id = lp.sample(&last_token_logits)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi tính toán Sampling: {}", e)))?;

        // 🚨 GIẢI MÃ LƯỢNG TỬ (DECODING)
        let decoded_text = self.tokenizer.decode(&[token_id], true)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi giải mã Tokenizer: {}", e)))?;

        Ok((token_id, decoded_text))
    }

    // 🚨 THÊM LẠI MẢNH GHÉP NÀY VÀO ĐÂY: HÀM QUẢN LÝ RÁC (GARBAGE COLLECTION)
    fn purge_vram(&self) -> Result<()> {
        info!("🧹 [Candle Bridge] Hệ thống rác tự động của Rust sẽ dọn dẹp Tensor cô lập. RAM an toàn.");
        Ok(())
    }
}