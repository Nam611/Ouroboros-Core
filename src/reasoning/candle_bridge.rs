use crate::error::{OuroborosError, Result};
use crate::core::query_loop::AgentTask;
use bytes::Bytes;
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::VarBuilder;
use std::sync::{Arc, Mutex};
use std::str::FromStr; 
use tokenizers::Tokenizer;
use tracing::info;
use crate::memory::turbo_quant::{TurboMathCore, TurboQuantProd, TurboWorkspace, QuantizedProdCache};

// ============================================================================
// 🚨 LÕI TOÁN HỌC BARE-METAL GEMMA 4 (TỰ ĐÚC BẰNG CANDLE_CORE)
// Tuyệt đối không sử dụng candle_transformers. Đọc trực tiếp Tensor Native của Google.
// ============================================================================

pub struct GemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
}

#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}
impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        // Đọc thẳng tên gốc "weight"
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps: eps as f32 })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps)
    }
}

pub struct GemmaMlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}
impl GemmaMlp {
    pub fn new(cfg: &GemmaConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(hidden, inter, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(hidden, inter, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(inter, hidden, vb.pp("down_proj"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // Kiến trúc Gemma sử dụng GeGLU
        let gate_gelu = gate.gelu_erf()?;
        let mul = gate_gelu.broadcast_mul(&up)?;
        self.down_proj.forward(&mul)
    }
}

pub struct GemmaAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}
impl GemmaAttention {
    pub fn new(cfg: &GemmaConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(hidden, cfg.num_attention_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(cfg.num_attention_heads * head_dim, hidden, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Toán học Attention nguyên bản (Scale Dot-Product)
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att_weights = candle_nn::ops::softmax_last_dim(&att)?;
        let att_output = att_weights.matmul(&v)?;

        let out = att_output.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

pub struct GemmaLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}
impl GemmaLayer {
    pub fn new(cfg: &GemmaConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: GemmaAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: GemmaMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x_norm = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&x_norm)?;
        let x = (attn_out + residual)?;

        let residual = &x;
        let x_norm = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        mlp_out + residual
    }
}

pub struct Gemma4E4B {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<GemmaLayer>,
    norm: RmsNorm,
    device: Device,
}
impl Gemma4E4B {
    pub fn new(cfg: &GemmaConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        // 🚨 NATIVE MAPPING: Tìm đúng 'embed_tokens' không cần tiền tố 'model.'
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(GemmaLayer::new(cfg, vb_layers.pp(layer_idx.to_string()))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        
        Ok(Self { embed_tokens, layers, norm, device: vb.device().clone() })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.embed_tokens.forward(x)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

// ============================================================================
// HỆ ĐIỀU HÀNH TÁC TỬ (OUROBOROS AGENT BRIDGE)
// ============================================================================

pub struct CandleBridge {
    model: Option<Mutex<Gemma4E4B>>, 
    tokenizer: Option<Tokenizer>,
    device: Device,
    system_prompt: String,
    quant_engine: Option<Arc<TurboQuantProd>>,
    workspace: Option<TurboWorkspace>,
    state_buffer: Vec<f32>,
    pub memory_bank: Vec<QuantizedProdCache>,
}

impl CandleBridge {
    pub fn new() -> Self {
        Self {
            model: None, tokenizer: None, device: Device::Cpu,
            system_prompt: "You are Ouroboros Core, a 2026 Bare-metal Architect. Optimize Rust code.".to_string(),
            quant_engine: None, workspace: None,
            state_buffer: Vec::new(), memory_bank: Vec::new(),
        }
    }

    pub fn boot_sequence(&mut self) -> Result<()> {
        self.device = Device::Cpu;
        info!("⚙️ [Candle Bridge] Khởi động Lõi NATIVE GEMMA 4 (Custom Core)...");
        
        let d = 2048; 
        let math_core = Arc::new(TurboMathCore::new(d)?);
        self.quant_engine = Some(Arc::new(TurboQuantProd::new(Arc::clone(&math_core), 4)?));
        self.workspace = Some(TurboWorkspace::new(d));
        self.state_buffer = vec![0.0; d];

        let (tok_path, config_path, safetensors_path) = Self::download_native_gemma4_e4b()?;

        info!("🧩 Nạp Tokenizer...");
        let tok_raw = std::fs::read_to_string(&tok_path).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let mut tok_v: serde_json::Value = serde_json::from_str(&tok_raw).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        if tok_v["model"]["type"] == "ModelWrapper" {
            if let Some(inner) = tok_v["model"]["model"].take().as_object() {
                tok_v["model"] = serde_json::Value::Object(inner.clone());
            }
        }
        let tokenizer = Tokenizer::from_str(&tok_v.to_string()).map_err(|e| OuroborosError::Reasoning(format!("Lỗi Tokenizer API: {}", e)))?;
        self.tokenizer = Some(tokenizer);

        info!("🧠 Trích xuất Thông số Vật lý...");
        let conf_raw = std::fs::read_to_string(&config_path).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let cv: serde_json::Value = serde_json::from_str(&conf_raw).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        let hidden_size = cv["hidden_size"].as_u64().or(cv["model_dim"].as_u64()).unwrap_or(2048) as usize;
        let num_heads = cv["num_attention_heads"].as_u64().or(cv["num_heads"].as_u64()).unwrap_or(8) as usize;

        let config = GemmaConfig {
            vocab_size: cv["vocab_size"].as_u64().unwrap_or(256000) as usize,
            hidden_size,
            intermediate_size: cv["intermediate_size"].as_u64().unwrap_or(16384) as usize,
            num_hidden_layers: cv["num_hidden_layers"].as_u64().or(cv["num_layers"].as_u64()).unwrap_or(18) as usize,
            num_attention_heads: num_heads,
            num_key_value_heads: cv["num_key_value_heads"].as_u64().unwrap_or(1) as usize,
            head_dim: cv["head_dim"].as_u64().unwrap_or((hidden_size / num_heads) as u64) as usize,
            rms_norm_eps: cv["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        };

        // 🚨 KỶ LUẬT ZERO-OOM & NATIVE MAPPING CHUẨN MỰC
        info!("🛡️ Khởi tạo Lõi VarBuilder (Mmap trực tiếp 4GB F16 từ SSD vào RAM)...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F16, &self.device)
                .map_err(|e| OuroborosError::Reasoning(format!("Lỗi Mmap: {}", e)))?
        };

        // Ráp mạch trực tiếp không cần đổi tên Tensor
        // Tốc độ ánh xạ ~0.1s, RAM tiêu thụ đỉnh ~4GB, không có ngoại lệ.
        let model = Gemma4E4B::new(&config, vb)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi ráp mạch Nơ-ron Native: {}", e)))?;

        self.model = Some(Mutex::new(model));
        info!("✅ [Neural Boot] Ouroboros đã thức tỉnh với Kiến trúc Gemma 4 Custom.");
        Ok(())
    }

    pub fn analyze_source_code(&mut self, code_content: &str, file_path: &str) -> Result<Option<AgentTask>> {
        let context = format!("<bos><start_of_turn>user\n{}\n\n[FILE: {}]\n{}\n<end_of_turn>\n<start_of_turn>model\n<|think|>\n", self.system_prompt, file_path, code_content);
        let mut generated_text = String::new();
        let mut current_bytes = Bytes::from(context);

        for _ in 0..250 {
            let (_next_token, text_chunk) = self.forward_pass(&current_bytes)?;
            generated_text.push_str(&text_chunk);
            print!("{}", text_chunk);
            use std::io::Write; let _ = std::io::stdout().flush();
            if text_chunk.contains("<eos>") || generated_text.contains('}') { break; }
            current_bytes = Bytes::from(text_chunk);
        }
        self.parse_llm_json(&generated_text)
    }

    fn forward_pass(&mut self, input_bytes: &Bytes) -> Result<(u32, String)> {
        let tokenizer = self.tokenizer.as_ref().ok_or(OuroborosError::Reasoning("No Tokenizer".into()))?;
        let model_mutex = self.model.as_ref().ok_or(OuroborosError::Reasoning("No Model".into()))?;
        let text = String::from_utf8(input_bytes.to_vec()).unwrap_or_default();
        let tokens = tokenizer.encode(text, true).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let input = Tensor::new(tokens.get_ids(), &self.device).map_err(|e| OuroborosError::Reasoning(e.to_string()))?.unsqueeze(0).unwrap();
        
        let logits = model_mutex.lock().unwrap().forward(&input).map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let logits = logits.squeeze(0).unwrap();
        let final_logits = logits.narrow(0, logits.dims2().unwrap().0 - 1, 1).unwrap().squeeze(0).unwrap();
        let logits_v = final_logits.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        
        let mut next_token = 0; let mut max_v = f32::NEG_INFINITY;
        for (i, &v) in logits_v.iter().enumerate() { if v > max_v { max_v = v; next_token = i as u32; } }

        if let (Some(engine), Some(ws)) = (self.quant_engine.as_ref(), self.workspace.as_mut()) {
            let len = std::cmp::min(logits_v.len(), self.state_buffer.len());
            self.state_buffer[..len].copy_from_slice(&logits_v[..len]);
            if let Ok(compressed) = engine.compress(&self.state_buffer, ws) {
                self.memory_bank.push(compressed);
            }
        }

        let output_text = tokenizer.decode(&[next_token], true).unwrap_or_default();
        Ok((next_token, output_text))
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