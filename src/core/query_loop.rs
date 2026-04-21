use crate::error::Result;
use crate::memory::semantic_graph::SemanticGraph;
use crate::reasoning::embedding_engine::EmbeddingEngine;
use crate::swarm::coordinator::SwarmCoordinator;
use tracing::{info, debug, warn};

// 🚨 DATA CONTRACT: Cấu trúc đại diện cho một Nhiệm vụ đột biến
#[derive(Debug, Clone)]
pub struct AgentTask {
    pub target_file: String,
    pub target_function: String,
    pub instruction: String, // Instruction này chính là chuỗi JSON do Gemma sinh ra
}

/// 🦂 [SWARM COORDINATOR HUB]
/// Trái tim điều phối các tác vụ đột biến vật lý.
/// Kiến trúc đã được tinh lọc: Không còn chứa não bộ (LLM) nội bộ. 
/// LLM giờ đây nằm ngoài (EvolutionDaemon), QueryLoop chỉ nhận JSON và phân phát cho Bầy đàn.
pub struct QueryLoop {
    pub semantic_graph: SemanticGraph,
    pub embedding_engine: EmbeddingEngine,
    swarm: SwarmCoordinator,
}

impl QueryLoop {
    pub fn new() -> Self {
        let embedding_engine = EmbeddingEngine::new().expect("❌ Lỗi khởi tạo Embedding Engine");
        let dim = embedding_engine.dimension;
        
        Self {
            semantic_graph: SemanticGraph::new(dim),
            embedding_engine,
            swarm: SwarmCoordinator::new(4), // Khởi tạo bầy đàn 4 luồng
        }
    }

    /// Kích hoạt Bầy đàn để thực thi hợp đồng do Gemma (EvolutionDaemon) ban hành
    pub async fn run(&mut self, task: AgentTask) -> Result<()> {
        info!("🦂 [Swarm Hub] Tiếp nhận Hợp đồng Đột biến từ Mạng Nơ-ron.");
        debug!("Tệp mục tiêu: {}", task.target_file);
        
        // Truyền thẳng chỉ thị (JSON) từ Gemma vào Swarm Coordinator
        match self.swarm.dispatch_task(&task.instruction).await {
            Ok(msg) => {
                info!("🎯 [SWARM SUCCESS] {}", msg);
                Ok(())
            },
            Err(e) => {
                warn!("⚔️ [SWARM REJECT] Bầy đàn đã từ chối hoặc thất bại khi sửa đổi: {}", e);
                // Ở tương lai, nếu lỗi, chúng ta có thể trả kết quả về cho Gemma để nó sửa lại (Self-Healing)
                Ok(()) // Trả về Ok để vòng lặp tiến hóa không bị sập hoàn toàn
            }
        }
    }
}