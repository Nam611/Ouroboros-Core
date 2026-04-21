use crate::error::{OuroborosError, Result};
use crate::core::query_loop::AgentTask;
use bytes::Bytes;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model as NativeGemma}; 
use std::sync::{Arc, Mutex};
use std::str::FromStr; 
use tokenizers::Tokenizer;
use tracing::info;
use std::io::{Read, Write};
use crate::memory::turbo_quant::{TurboMathCore, TurboQuantProd, TurboWorkspace, QuantizedProdCache};

pub struct CandleBridge {
    model: Option<Mutex<NativeGemma>>, 
    tokenizer: Option<Tokenizer>,
    device: Device,
    system_prompt: String,
    pub seqlen_offset: usize, // 🚨 CỐT LÕI VẬT LÝ: Con trỏ định vị KV Cache
    quant_engine: Option<Arc<TurboQuantProd>>,
    workspace: Option<TurboWorkspace>,
    state_buffer: Vec<f32>,
    pub memory_bank: Vec<QuantizedProdCache>,
}

impl CandleBridge {
    pub fn new() -> Self {
        Self {
            model: None, tokenizer: None, device: Device::Cpu,
            system_prompt: "You are Ouroboros Core, a 2026 Bare-metal Architect. Optimize Rust code. ONLY OUTPUT JSON.".to_string(),
            seqlen_offset: 0, quant_engine: None, workspace: None,
            state_buffer: Vec::new(), memory_bank: Vec::new(),
        }
    }

    pub fn boot_sequence(&mut self) -> Result<()> {
        self.device = Device::Cpu;
        info!("⚙️ [Candle Bridge] Khởi động Lõi Gemma 4 NATIVE (Hardware-Accelerated KV Cache)...");
        
        let d = 2048; 
        let math_core = Arc::new(TurboMathCore::new(d)?);
        self.quant_engine = Some(Arc::new(TurboQuantProd::new(Arc::clone(&math_core), 4)?));
        self.workspace = Some(TurboWorkspace::new(d));
        self.state_buffer = vec![0.0; d];

        let (tok_path, config_path, safetensors_path) = Self::download_native_gemma4_e4b()?;

        info!("🧩 Nạp Tokenizer...");
        let tok_raw = std::fs::read_to_string(&tok_path).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
        let mut tok_v: serde_json::Value = serde_json::from_str(&tok_raw).map_err(|e: serde_json::Error| OuroborosError::Reasoning(e.to_string()))?;
        if tok_v["model"]["type"] == "ModelWrapper" {
            if let Some(inner) = tok_v["model"]["model"].take().as_object() {
                tok_v["model"] = serde_json::Value::Object(inner.clone());
            }
        }
        let tokenizer = Tokenizer::from_str(&tok_v.to_string()).map_err(|e: Box<dyn std::error::Error + Send + Sync>| OuroborosError::Reasoning(format!("Lỗi Tokenizer API: {}", e)))?;
        self.tokenizer = Some(tokenizer);

        info!("🧠 Khởi tạo Schema Gemma chuẩn...");
        let conf_raw = std::fs::read_to_string(&config_path).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
        let cv: serde_json::Value = serde_json::from_str(&conf_raw).map_err(|e: serde_json::Error| OuroborosError::Reasoning(e.to_string()))?;

        let hidden_size = cv["hidden_size"].as_u64().or(cv["model_dim"].as_u64()).unwrap_or(2048) as usize;
        let num_heads = cv["num_attention_heads"].as_u64().or(cv["num_heads"].as_u64()).unwrap_or(8) as usize;

        let config = Config {
            vocab_size: cv["vocab_size"].as_u64().unwrap_or(256000) as usize,
            hidden_size,
            intermediate_size: cv["intermediate_size"].as_u64().unwrap_or(16384) as usize,
            num_hidden_layers: cv["num_hidden_layers"].as_u64().or(cv["num_layers"].as_u64()).unwrap_or(18) as usize,
            num_attention_heads: num_heads,
            num_key_value_heads: cv["num_key_value_heads"].as_u64().unwrap_or(1) as usize,
            head_dim: cv["head_dim"].as_u64().unwrap_or((hidden_size / num_heads) as u64) as usize,
            rms_norm_eps: cv["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: cv["rope_theta"].as_f64().unwrap_or(10000.0),
            attention_bias: cv["attention_bias"].as_bool().unwrap_or(false),
            hidden_act: Some(candle_nn::Activation::Gelu),
            hidden_activation: Some(candle_nn::Activation::Gelu),
            max_position_embeddings: cv["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
        };

        let remapped_path = safetensors_path.with_file_name("model_remapped.safetensors");
        if !remapped_path.exists() {
            info!("🛠️ Kích hoạt Lò rèn Safetensors (Zero-OOM File Stream)...");
            let mut old_file = std::fs::File::open(&safetensors_path).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
            let mut header_size_buf = [0u8; 8];
            old_file.read_exact(&mut header_size_buf).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
            let header_size = u64::from_le_bytes(header_size_buf) as usize;
            
            let mut header_buf = vec![0u8; header_size];
            old_file.read_exact(&mut header_buf).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
            let mut json: serde_json::Value = serde_json::from_str(&String::from_utf8(header_buf).unwrap()).unwrap();

            if let Some(obj) = json.as_object_mut() {
                let mut new_keys = std::collections::HashMap::new();
                let keys: Vec<String> = obj.keys().cloned().collect();
                for k in keys {
                    if k == "__metadata__" { continue; }
                    let mut new_k = k.clone();
                    if new_k.starts_with("transformer.") { new_k = new_k.replace("transformer.", "model."); }
                    if !new_k.starts_with("model.") && !new_k.starts_with("lm_head") && !new_k.starts_with("vocab") { new_k = format!("model.{}", new_k); }
                    if new_k != k { new_keys.insert(k, new_k); }
                }
                for (old_k, new_k) in new_keys { if let Some(val) = obj.remove(&old_k) { obj.insert(new_k, val); } }
            }

            let new_header_bytes = serde_json::to_vec(&json).unwrap();
            let new_header_size = new_header_bytes.len() as u64;
            
            let mut new_file = std::fs::File::create(&remapped_path).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
            new_file.write_all(&new_header_size.to_le_bytes()).unwrap();
            new_file.write_all(&new_header_bytes).unwrap();
            std::io::copy(&mut old_file, &mut new_file).map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?;
        }

        info!("🛡️ Ánh xạ Safetensors Lazily (F16 - Mmap)...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[remapped_path], DType::F16, &self.device)
                .map_err(|e: candle_core::Error| OuroborosError::Reasoning(format!("Lỗi nạp Mmap: {}", e)))?
        };

        // 🚨 KHÔI PHỤC KIẾN TRÚC GỐC ĐỂ MỞ KHÓA KV CACHE
        let model = NativeGemma::new(false, &config, vb)
            .map_err(|e: candle_core::Error| OuroborosError::Reasoning(format!("Lỗi ráp mạch Nơ-ron: {}", e)))?;

        self.model = Some(Mutex::new(model));
        info!("✅ [Neural Boot] Ouroboros đã thức tỉnh. Sẵn sàng ép xung Autoregressive.");
        Ok(())
    }

    /// 🚀 THUẬT TOÁN SUY LUẬN BARE-METAL (AUTOREGRESSIVE O(1) BẰNG KV CACHE)
    pub fn analyze_source_code(&mut self, code_content: &str, file_path: &str) -> Result<Option<AgentTask>> {
        self.seqlen_offset = 0; // Reset trí nhớ khi bắt đầu file mới
        let context = format!("<bos><start_of_turn>user\n{}\n\n[FILE: {}]\n{}\n<end_of_turn>\n<start_of_turn>model\n<|think|>\n", self.system_prompt, file_path, code_content);
        
        let tokenizer = self.tokenizer.as_ref().ok_or(OuroborosError::Reasoning("No Tokenizer".into()))?;
        let model_mutex = self.model.as_ref().ok_or(OuroborosError::Reasoning("No Model".into()))?;

        // 🚨 TỐI ƯU TOÁN HỌC: Chỉ băm từ khóa đúng 1 lần duy nhất
        let mut next_tokens = tokenizer.encode(context, true)
            .map_err(|e: Box<dyn std::error::Error + Send + Sync>| OuroborosError::Reasoning(e.to_string()))?
            .get_ids().to_vec();
            
        let mut generated_text = String::new();

        for _ in 0..250 {
            // Nạp Tensor (Vòng 1: Dài. Vòng 2+: Đúng 1 token)
            let input_tensor = Tensor::new(next_tokens.as_slice(), &self.device)
                .map_err(|e: candle_core::Error| OuroborosError::Reasoning(e.to_string()))?
                .unsqueeze(0).unwrap();
            
            // Model tính toán và tự động nạp trạng thái vào KV Cache
            let logits = model_mutex.lock().unwrap().forward(&input_tensor, self.seqlen_offset)
                .map_err(|e: candle_core::Error| OuroborosError::Reasoning(e.to_string()))?;
            
            // 🚨 TỊNH TIẾN CON TRỎ BỘ NHỚ
            self.seqlen_offset += next_tokens.len();

            let logits = logits.squeeze(0).unwrap();
            let final_logits = logits.narrow(0, logits.dims2().unwrap().0 - 1, 1).unwrap().squeeze(0).unwrap();
            let logits_v = final_logits.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
            
            let mut next_token = 0; let mut max_v = f32::NEG_INFINITY;
            for (i, &v) in logits_v.iter().enumerate() { if v > max_v { max_v = v; next_token = i as u32; } }

            // Lõi TurboQuant hút logits
            if let (Some(engine), Some(ws)) = (self.quant_engine.as_ref(), self.workspace.as_mut()) {
                let len = std::cmp::min(logits_v.len(), self.state_buffer.len());
                self.state_buffer[..len].copy_from_slice(&logits_v[..len]);
                if let Ok(compressed) = engine.compress(&self.state_buffer, ws) {
                    self.memory_bank.push(compressed);
                }
            }

            let text_chunk = tokenizer.decode(&[next_token], true).unwrap_or_default();
            generated_text.push_str(&text_chunk);
            print!("{}", text_chunk);
            use std::io::Write; let _ = std::io::stdout().flush();

            if text_chunk.contains("<eos>") || generated_text.contains('}') { break; }
            
            // 🚨 CHUẨN BỊ CHO VÒNG LẶP SAU: Chỉ nạp đúng 1 token mới vào mạng (O(1))
            next_tokens = vec![next_token]; 
        }
        self.parse_llm_json(&generated_text)
    }

    fn parse_llm_json(&self, raw: &str) -> Result<Option<AgentTask>> {
        let trimmed = if let Some(idx) = raw.find("</|think|>") { &raw[idx+10..] } else { raw }.trim();
        if let (Some(s), Some(e)) = (trimmed.find('{'), trimmed.rfind('}')) {
            Ok(Some(AgentTask { target_file: String::new(), target_function: "auto".into(), instruction: trimmed[s..=e].to_string() }))
        } else { Ok(None) }
    }

    fn download_native_gemma4_e4b() -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        let cache_dir = std::path::PathBuf::from(".ouroboros_cache/gemma4_native");
        if !cache_dir.exists() { std::fs::create_dir_all(&cache_dir).unwrap(); }
        let base_url = "https://huggingface.co/google/gemma-4-E4B/resolve/main";
        let files = vec![("tokenizer.json", cache_dir.join("tokenizer.json")), ("config.json", cache_dir.join("config.json")), ("model.safetensors", cache_dir.join("model.safetensors"))];
        for (name, path) in &files {
            if !path.exists() {
                info!("⬇️ Kéo vũ khí: {}", name);
                let mut cmd = std::process::Command::new("curl");
                cmd.args(["-f", "-L", "-#", "-o", path.to_str().unwrap()]);
                let token = std::env::var("HF_TOKEN").unwrap_or_default();
                if !token.is_empty() { cmd.arg("-H").arg(format!("Authorization: Bearer {}", token)); }
                let url = format!("{}/{}", base_url, name);
                if !cmd.arg(url).status().unwrap().success() { return Err(OuroborosError::Reasoning("Download failed".into())); }
            }
        }
        Ok((files[0].1.clone(), files[1].1.clone(), files[2].1.clone()))
    }
}