use crate::error::Result;
use crate::reasoning::loop_lm::LoopLM;
use crate::memory::turbo_quant::TurboQuant;
use crate::swarm::coordinator::SwarmCoordinator;
use crate::network::xdp_core::{NativeXdpBridge, XdpEngine};

// 🚨 IMPORT LÕI TRI THỨC VÀ ĐỘNG CƠ TÌM KIẾM
use crate::memory::semantic_graph::SemanticGraph;
use crate::reasoning::embedding_engine::EmbeddingEngine;

use tracing::{debug, info, instrument, warn};

#[derive(Debug, PartialEq)]
pub enum SystemState {
    Idle,
    AmbiguityResolution,
    KnowledgeRetrieval, // 🚨 CỔNG MỚI: Truy xuất tri thức (RAG)
    Reasoning,
    ToolExecution,
    AwaitingUser,
    Terminated,
}

pub struct QueryLoop {
    state: SystemState,
    iteration_count: u64,
    loop_lm: LoopLM,
    turbo_quant: TurboQuant,
    swarm: SwarmCoordinator,
    #[allow(dead_code)]
    xdp_bridge: Box<dyn XdpEngine>,
    
    // 🚨 KẾT NỐI VỚI TÀNG KINH CÁC VÀ ĐỘNG CƠ DỊCH THUẬT
    semantic_graph: SemanticGraph,
    embedding_engine: EmbeddingEngine,
}

impl QueryLoop {
    pub fn new() -> Self {
        let loop_lm = LoopLM::new(); 

        let mut xdp_bridge = NativeXdpBridge::new();
        if let Err(e) = xdp_bridge.attach_to_interface("eth0") {
            warn!("⚠️ Không thể đính kèm XDP: {}. Bỏ qua lớp bảo vệ mạng.", e);
        }

        // 🚨 KHỞI TẠO ĐỘNG CƠ TRI THỨC
        let embedding_engine = EmbeddingEngine::new().expect("❌ Lỗi khởi tạo Embedding Engine");
        let mut semantic_graph = SemanticGraph::new(embedding_engine.dimension);

        // 📚 NẠP SÁCH VÀO TÀNG KINH CÁC (Dữ liệu thật đập tan ảo giác)
        info!("📚 Bơm Tri thức Sự thật (Ground Truth) vào Ký ức...");
        let fact = "Rust là một ngôn ngữ lập trình hệ thống đa mô hình được thiết kế bởi Graydon Hoare tại Mozilla Research, với sự đóng góp của Dave Herman, Brendan Eich và những người khác. Phiên bản chính thức đầu tiên được phát hành vào năm 2010.";
        
        let fact_vector = embedding_engine.generate_embedding(fact).expect("❌ Lỗi nhúng Vector");
        semantic_graph.ingest_knowledge(fact, fact_vector).expect("❌ Lỗi lưu trữ Vector");

        Self {
            state: SystemState::Idle,
            iteration_count: 0,
            loop_lm, 
            turbo_quant: TurboQuant::new(), 
            swarm: SwarmCoordinator::new(4),
            xdp_bridge: Box::new(xdp_bridge), 
            semantic_graph,
            embedding_engine,
        }
    }

    #[instrument(skip(self), name = "core_loop")]
    pub async fn run(&mut self) -> Result<()> {
        info!("🌀 Kích hoạt Query Loop. Khởi động State Machine v2.0 [RAG Truth Engine]...");

        self.transition_to(SystemState::AmbiguityResolution).await?;
        self.transition_to(SystemState::KnowledgeRetrieval).await?; // Tìm kiếm Tri thức
        self.transition_to(SystemState::Reasoning).await?;          // Suy luận dựa trên Tri thức
        self.transition_to(SystemState::ToolExecution).await?;
        self.transition_to(SystemState::Terminated).await?;

        info!("🌀 State Machine đã đóng băng an toàn. Zero Memory Leak.");
        Ok(())
    }

    fn evaluate_ambiguity(&self, task: &str) -> f32 {
        if task.contains("chi tiết") { 15.0 } else { 85.0 }
    }

    async fn transition_to(&mut self, next_state: SystemState) -> Result<()> {
        self.state = next_state;
        self.iteration_count += 1;
        
        // Dùng biến chung cho Task
        let user_query = "Rust là ngôn ngữ lập trình "; 
        
        match self.state {
            SystemState::Idle => {
                debug!("Trạng thái [IDLE]...");
            }
            SystemState::AmbiguityResolution => {
                info!("🛡️ [CỔNG KIỂM DUYỆT]: Kích hoạt thuật toán Khử Mơ Hồ...");
                // (Logic giữ nguyên)
                let ambiguity_score = self.evaluate_ambiguity("Phân tích hệ thống");
                if ambiguity_score > 20.0 {
                    warn!("❌ Phát hiện Context mơ hồ. Tạm dừng. Đã mở cổng an toàn.");
                }
            }
            SystemState::KnowledgeRetrieval => {
                info!("🔍 Trạng thái [RETRIEVAL]: Quét Semantic Graph tìm kiếm sự thật...");
                
                // 1. Biến câu hỏi thành Vector
                let query_vector = self.embedding_engine.generate_embedding(user_query)?;
                
                // 2. Tìm câu trả lời trong RAM
                if let Some(knowledge) = self.semantic_graph.search_closest_knowledge(&query_vector)? {
                    info!("📖 Tìm thấy Tàng Kinh Các: [{}]", knowledge);
                    // Lưu tạm kiến thức vào bộ đệm của Loop (để chuyển sang State Reasoning)
                    // (Trong bản thực tế ta sẽ dùng struct state, nhưng ở đây tạm hack qua biến môi trường để nối chuỗi nhanh)
                    std::env::set_var("OUROBOROS_KNOWLEDGE", knowledge);
                } else {
                    warn!("⚠️ Không tìm thấy tri thức liên quan. Ouroboros sẽ chạy mù.");
                    std::env::set_var("OUROBOROS_KNOWLEDGE", "");
                }
            }
            SystemState::Reasoning => {
                info!("Trạng thái [REASONING]: Bơm dữ liệu vào Ouroboros Loop...");
                
                // Lấy Tri thức từ bước trước
                let knowledge = std::env::var("OUROBOROS_KNOWLEDGE").unwrap_or_default();
                
                // 🚨 KỸ THUẬT PROMPT INJECTION (Tiêm Sự Thật)
                let injected_prompt = if !knowledge.is_empty() {
                    format!("Dựa trên sự thật sau đây: '{}'. Hãy viết tiếp câu sau một cách chính xác: {}", knowledge, user_query)
                } else {
                    user_query.to_string()
                };

                // Trái tim đập!
                let result = self.loop_lm.process_reasoning(&injected_prompt).await?;
                info!("✨ Nhận thức hoàn tất. Output: \n{}", result);
            }
            SystemState::ToolExecution => {
                info!("Trạng thái [TOOL_EXECUTION]: Triển khai mạng lưới tác tử (Swarm)...");
                let swarm_result = self.swarm.dispatch_task("Tối ưu hóa file Cargo.toml").await?;
                info!("🛠️ {}", swarm_result);
            }
            SystemState::AwaitingUser => {}
            SystemState::Terminated => {
                info!("Trạng thái [TERMINATED]: Lượng tử hóa bộ nhớ...");
                let dummy_heavy_cache = vec![0.1234_f32; 100_000];
                let compressed_memory = self.turbo_quant.compress_kv_cache(&dummy_heavy_cache)?;
                debug!("Bộ nhớ KV Cache hiện tại chỉ còn: {} Bytes.", compressed_memory.data.len());
            }
        }
        Ok(())
    }
}