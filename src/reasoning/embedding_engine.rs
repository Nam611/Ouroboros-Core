use crate::error::{OuroborosError, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, HiddenAct};
use tokenizers::{Tokenizer, PaddingParams};
use tracing::{debug, info};

/// 🌌 LÕI EMBEDDING: Trạm Biến Áp Ngôn Ngữ -> Vector (all-MiniLM-L6-v2)
pub struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pub dimension: usize,
}

impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        info!("🌌 [Embedding Engine] Kích hoạt buồng đốt Vector trên: {:?}", device);

        info!("⬇️ Kéo Lõi all-MiniLM-L6-v2 từ không gian mạng...");
        let (config_path, tokenizer_path, weights_path) = Self::download_minilm_core()?;

        info!("🧩 Nạp Tokenizer Đa chiều...");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi Tokenizer: {}", e)))?;
        
        // Cấu hình Padding để xử lý câu văn có độ dài khác nhau
        // 🚨 BẢN VÁ: get_padding_mut() đã trả về &mut, không cần thêm mut ở biến
        if let Some(pad_params) = tokenizer.get_padding_mut() {
            pad_params.pad_token = String::from("[PAD]");
        } else {
            tokenizer.with_padding(Some(PaddingParams {
                pad_token: String::from("[PAD]"),
                ..Default::default()
            }));
        }

        info!("🧠 Ráp Mạch BERT Transformer...");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let mut config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        
        // MiniLM dùng hàm kích hoạt GELU
        config.hidden_act = HiddenAct::Gelu;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], candle_core::DType::F32, &device)
                .map_err(|e| OuroborosError::Reasoning(format!("Lỗi Mmap: {}", e)))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi tải Model: {}", e)))?;

        info!("✅ [Embedding Engine] Đã sẵn sàng nén tri thức thành Vector 384 chiều.");

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension: 384, // Cố định vật lý của MiniLM-L6-v2
        })
    }

    /// 🧮 LÕI TOÁN HỌC: Chuyển Chữ -> Mảng 384 con số
    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // 1. Tokenize (Cắt chữ)
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        let token_ids = tokens.get_ids();
        
        let token_tensor = Tensor::new(token_ids, &self.device)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?
            .unsqueeze(0) // Thêm chiều Batch (1, Seq_len)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        // 2. Loại bỏ nhiễu bằng Attention Mask (Chỉ tập trung vào chữ thật, bỏ qua khoảng trắng PAD)
        let token_type_ids = token_tensor.zeros_like()
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        // 3. Đẩy qua Mạng Nơ-ron BERT
        // 🚨 BẢN VÁ: Cung cấp tham số `attention_mask` là None cho single-batch
        let embeddings = self.model.forward(&token_tensor, &token_type_ids, None)
            .map_err(|e| OuroborosError::Reasoning(format!("Lỗi Forward BERT: {}", e)))?;

        // 4. KỶ LUẬT MEAN POOLING (Lấy trung bình cộng các vector của từng chữ)
        // Ép trận Tensor 3D (Batch, Seq, 384) thành 2D (Batch, 384)
        let (_batch_size, seq_len, _hidden_size) = embeddings.dims3()
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        
        let sum_embeddings = embeddings.sum(1)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
        
        let mean_embeddings = (sum_embeddings / (seq_len as f64))
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        // 5. L2 NORMALIZATION (Chuẩn hóa độ dài Vector về 1 để tối ưu Cosine Similarity)
        let l2_norm = mean_embeddings.sqr()
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?
            .sqrt()
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;
            
        let normalized = mean_embeddings.broadcast_div(&l2_norm)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        // 6. Rút dữ liệu từ VRAM ra RAM để lưu vào Semantic Graph
        let vector_f32 = normalized.squeeze(0)
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| OuroborosError::Reasoning(e.to_string()))?;

        Ok(vector_f32)
    }

    /// Động cơ I/O: Tải model all-MiniLM-L6-v2 (Zero-Proxy)
    fn download_minilm_core() -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
        let cache_dir = std::path::PathBuf::from(".ouroboros_cache/minilm");
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| OuroborosError::Reasoning(format!("Lỗi tạo thư mục: {}", e)))?;
        }

        let base_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main";
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        let mut paths = vec![];

        for file in files {
            let file_path = cache_dir.join(file);
            paths.push(file_path.clone());
            
            if file_path.exists() {
                debug!("✅ Lõi MiniLM Local: {}", file);
                continue;
            }
            
            info!("⬇️ Kéo [{}]...", file);
            let url = format!("{}/{}", base_url, file);
            
            let status = std::process::Command::new("curl")
                .args(["-L", "-#", "-o", file_path.to_str().unwrap(), &url])
                .env("NO_PROXY", "*").env("no_proxy", "*")
                .status()
                .map_err(|e| OuroborosError::Reasoning(format!("OS từ chối lệnh cURL: {}", e)))?;
                
            if !status.success() {
                let _ = std::fs::remove_file(&file_path);
                return Err(OuroborosError::Reasoning(format!("cURL thất bại: {}", file)));
            }
        }
        Ok((paths[0].clone(), paths[1].clone(), paths[2].clone()))
    }
}