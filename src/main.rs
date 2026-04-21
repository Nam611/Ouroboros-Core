pub mod core;
pub mod error;
pub mod memory;
pub mod network;
pub mod reasoning;
pub mod swarm;

use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use crate::core::query_loop::AgentTask;
use crate::memory::indexer::ProjectScanner;

const INDEX_FILE_PATH: &str = ".ouroboros_index.bin";

/// 🛡️ OUROBOROS TITANIUM OS - Agentic Execution Core
#[derive(Parser)]
#[command(name = "ouroboros")]
#[command(about = "Bare-metal Agentic OS for Software Engineering", long_about = None)]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 🧬 Đột biến mã nguồn (Tối ưu hóa, sửa lỗi, viết tính năng mới)
    Mutate {
        #[arg(short, long)]
        file: String,
        #[arg(long)]
        func: String,
        #[arg(short, long)]
        goal: String,
    },
    
    /// 📚 Quét toàn bộ dự án và nạp vào Semantic Graph (Phase 2)
    Index {
        #[arg(short, long)]
        dir: String,
    },

    /// 💥 Khai hỏa tĩnh: Test cường độ cao Lõi Lượng tử Toán học (Phase 2)
    TestQuant {
        #[arg(short, long, default_value_t = 2048)]
        size: usize,
    },

    /// 🌌 [PHASE 5] ĐIỂM KỲ DỊ: Kích hoạt Động cơ Tiến hóa Tự trị (Gemma Core)
    Evolve,

    /// 🧠 Khai hỏa tĩnh: Test Lõi Nhận thức Gemma E4B + TurboQuant (Phase 5)
    TestBrain, // ✅ ĐÂY LÀ VỊ TRÍ CHUẨN XÁC ĐỘC LẬP!
}

// 🚨 BẢO TOÀN QUYỀN KIỂM SOÁT VẬT LÝ
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // TỊNH HÓA OS (Khởi động lạnh)
    let toxic_vars = [
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", 
        "all_proxy", "ALL_PROXY", "HF_ENDPOINT"
    ];
    for var in toxic_vars { std::env::remove_var(var); }
    std::env::set_var("HF_ENDPOINT", "https://huggingface.co");
    std::env::set_var("NO_PROXY", "*");

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::DEBUG.into()))
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    tracing::info!("🛡️ [Titanium Protocol] Khởi động lạnh hoàn tất.");

    // ĐÚC ĐỘNG CƠ BẤT ĐỒNG BỘ ĐA LUỒNG
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        let mut ouroboros = core::query_loop::QueryLoop::new();
        
        // KHÔI PHỤC KÝ ỨC TỪ ĐĨA TỪ
        if std::path::Path::new(INDEX_FILE_PATH).exists() {
            if let Ok(loaded_graph) = crate::memory::semantic_graph::SemanticGraph::load_from_disk(INDEX_FILE_PATH) {
                ouroboros.semantic_graph = loaded_graph;
            }
        }

        match cli.command {
            Commands::Mutate { file, func, goal } => {
                tracing::info!("🚀 Kích hoạt Mệnh lệnh Đột biến (Mutate) trên file: {}", file);
                let task = AgentTask { target_file: file, target_function: func, instruction: goal };
                if let Err(e) = ouroboros.run(task).await { tracing::error!("❌ Lỗi thực thi Mutate: {:?}", e); }
            },
            
            Commands::Index { dir } => {
                tracing::info!("🚀 Kích hoạt Chiến dịch Indexing trên thư mục: [{}]", dir);
                let scanner = ProjectScanner::new();
                match scanner.scan_directory(&dir).await {
                    Ok(files) => {
                        tracing::info!("🔥 Phát hiện {} file. Đang băm nhỏ và Nhúng Vector...", files.len());
                        let mut total_chunks = 0;
                        ouroboros.semantic_graph.clear();

                        for file_path in files {
                            if let Ok(content) = scanner.read_file_content(&file_path).await {
                                let chunks: Vec<String> = content.chars()
                                    .collect::<Vec<char>>().chunks(1000)
                                    .map(|c| c.iter().collect::<String>()).collect();

                                for (idx, chunk) in chunks.iter().enumerate() {
                                    let metadata_label = format!("File: {:?} (Phần {})", file_path, idx + 1);
                                    let combined_text = format!("{}\nCode:\n{}", metadata_label, chunk);
                                    if let Ok(vector) = ouroboros.embedding_engine.generate_embedding(&combined_text) {
                                        if let Err(e) = ouroboros.semantic_graph.ingest_knowledge(&combined_text, vector) {
                                            tracing::warn!("⚠️ Lỗi lưu chunk: {}", e);
                                        } else { total_chunks += 1; }
                                    }
                                }
                            }
                        }
                        tracing::info!("✅ Đã nạp {} khối tri thức vào RAM.", total_chunks);
                        if let Err(e) = ouroboros.semantic_graph.save_to_disk(INDEX_FILE_PATH) {
                            tracing::error!("❌ Thất bại khi đóng băng Tàng Kinh Các: {:?}", e);
                        }
                    },
                    Err(e) => tracing::error!("❌ Lỗi quét dự án: {:?}", e)
                }
            },

            Commands::TestQuant { size } => {
                let d = size; 
                
                tracing::info!("🚀 [STATIC FIRE] Kích hoạt Động cơ TurboQuant Prod (Zero-Allocation)...");
                tracing::info!("⚙️ Khởi tạo không gian {} chiều...", d);
                
                let math_core = std::sync::Arc::new(
                    crate::memory::turbo_quant::TurboMathCore::new(d)
                        .expect("❌ Lỗi khởi tạo Lõi Toán học!")
                );

                let target_bits = 4;
                let quantizer = crate::memory::turbo_quant::TurboQuantProd::new(std::sync::Arc::clone(&math_core), target_bits)
                    .expect("❌ Lỗi khởi tạo TurboQuantProd!");

                tracing::info!("🛡️ Cấp phát TurboWorkspace (Zero-Allocation Buffer)...");
                let mut workspace = crate::memory::turbo_quant::TurboWorkspace::new(d);

                let mut raw_data = vec![0.0; d];
                for i in 0..d { raw_data[i] = (i as f32).sin() * 2.5; }
                
                let mut norm_sq = 0.0;
                for &v in &raw_data { norm_sq += v * v; }
                let norm = norm_sq.sqrt();
                if norm > 0.0 {
                    for v in &mut raw_data { *v /= norm; }
                }

                tracing::info!("⚡ Đang nạp vào buồng nén lượng tử (Compressing)...");
                let start_compress = std::time::Instant::now();
                let q_cache = quantizer.compress(&raw_data, &mut workspace).expect("❌ Lỗi nén dữ liệu!");
                let compress_time = start_compress.elapsed();

                tracing::info!("🔓 Đang giải nén (Decompressing)...");
                let mut restored_data = vec![0.0; d];
                let start_decompress = std::time::Instant::now();
                quantizer.decompress_into(&q_cache, &mut workspace, &mut restored_data).expect("❌ Lỗi giải nén!");
                let decompress_time = start_decompress.elapsed();

                let mut original_ip = 0.0;
                let mut restored_ip = 0.0;
                for i in 0..d {
                    original_ip += raw_data[i] * raw_data[i];
                    restored_ip += raw_data[i] * restored_data[i];
                }

                tracing::info!("📊 BÁO CÁO VẬT LÝ LƯỢNG TỬ ({} Bits - {} Dimensions):", target_bits, d);
                tracing::info!("  - Inner Product Gốc:      {:.6}", original_ip);
                tracing::info!("  - Inner Product Phục hồi: {:.6}", restored_ip);
                tracing::info!("  - Độ lệch (Bias / Error): {:.6}", (original_ip - restored_ip).abs());
                tracing::info!("⏱️ THỜI GIAN ĐÁP ỨNG:");
                tracing::info!("  - Nén: {:?} | Giải nén: {:?}", compress_time, decompress_time);
            },

            Commands::Evolve => {
                tracing::info!("🌌 [THE SINGULARITY] Kích hoạt Vòng lặp Tiến hóa Tự trị...");
                let mut daemon = crate::core::evolution::EvolutionDaemon::new(".");
                
                if let Err(e) = daemon.ignite(&mut ouroboros).await {
                    tracing::error!("❌ Động cơ tiến hóa sụp đổ: {:?}", e);
                }
            },
            
            // 🚨 [PHASE 5] THE SINGULARITY: KHAI HỎA NÃO BỘ
            Commands::TestBrain => {
                tracing::info!("🚀 [STATIC FIRE] Kích hoạt Điểm Kỳ Dị (Gemma E4B + TurboQuant)...");
                
                let mut bridge = crate::reasoning::candle_bridge::CandleBridge::new();
                
                // Lệnh này sẽ kích hoạt CURL để tải 2.8GB GGUF
                tracing::info!("⬇️ Đang kết nối Trạm HuggingFace. Vui lòng chờ quá trình nạp dữ liệu vật lý...");
                bridge.boot_sequence().expect("❌ Lỗi khởi động Não bộ!");
                
                // Bơm đoạn code giả lập để ép Gemma suy nghĩ và kích hoạt TurboQuant
                let dummy_code = "fn calculate_fibonacci(n: u32) -> u32 {\n    if n <= 1 { return n; }\n    calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)\n}";
                tracing::info!("⚡ Bơm dữ liệu Sandbox. Ép Gemma suy luận và nén trí nhớ...");
                
                bridge.analyze_source_code(dummy_code, "src/math/fibonacci.rs")
                    .expect("❌ Lỗi suy luận cốt lõi!");
                
                tracing::info!("🏆 [HOÀN TẤT] Ca phẫu thuật The Singularity đã thành công!");
            },
        } 
    }); 
    Ok(())
}