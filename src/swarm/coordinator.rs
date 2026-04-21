use crate::error::{OuroborosError, Result};
use crate::swarm::tools::{AgentTools, FileMutationContract};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use std::time::Duration;

/// 🚨 GIAO THỨC TRUYỀN TIN BẦY ĐÀN (SWARM PROTOCOL)
#[derive(Debug, Clone)]
pub enum SwarmMessage {
    CodeGenerated { code: String },
    CompilerPassed,
    CompilerFailed { reason: String },
    SecurityApproved,
    SecurityViolated { reason: String },
}

pub struct SwarmCoordinator {
    #[allow(dead_code)]
    max_agents: usize,
}

impl SwarmCoordinator {
    pub fn new(max_agents: usize) -> Self {
        info!("🦂 [Swarm] Kích hoạt Buồng họp ảo Titanium. Giới hạn: {} luồng tác tử.", max_agents);
        Self { max_agents }
    }

    /// 🌪️ ĐIỀU PHỐI ĐA TÁC TỬ (CONCURRENT DISPATCH)
    pub async fn dispatch_task(&mut self, raw_llm_output: &str) -> Result<String> {
        let contract = self.extract_and_parse_contract(raw_llm_output)?;
        info!("🚀 Swarm tiếp nhận nhiệm vụ trên file: [{}]", contract.target_file);

        let (tx, mut rx) = mpsc::channel::<SwarmMessage>(32);

        // --- BƯỚC 1: KÍCH HOẠT AGENT ALPHA (THE CODER) ---
        let tx_alpha = tx.clone();
        let code_from_llm = contract.optimized_code.clone();
        
        tokio::spawn(async move {
            debug!("🐺 [Agent Alpha] Đang trích xuất mã nguồn từ luồng nhận thức...");
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx_alpha.send(SwarmMessage::CodeGenerated { code: code_from_llm }).await;
        });

        let mut consensus_count = 0;

        // --- BƯỚC 2: VÒNG LẶP QUẢN TRỊ ĐỒNG THUẬN ---
        while let Some(msg) = rx.recv().await {
            match msg {
                SwarmMessage::CodeGenerated { code } => {
                    info!("🐺 [Agent Alpha] Mã nguồn đã sẵn sàng. Kích hoạt Beta và Gamma thẩm định song song.");

                    // --- KÍCH HOẠT BETA (THE QA) ---
                    let tx_beta = tx.clone();
                    let code_for_beta = code.clone();
                    let target_file_name = contract.target_file.clone();
                    
                    tokio::spawn(async move {
                        debug!("🛡️ [Agent Beta] Đang khởi tạo Lồng ấp kiểm thử...");
                        let sandbox_file_path = format!(".ouroboros_sandbox/temp_run/{}", target_file_name);
                        
                        match crate::swarm::tools::TitaniumGate::create_sandbox(&sandbox_file_path, &code_for_beta).await {
                            Ok(workspace_dir) => {
                                match crate::swarm::tools::TitaniumGate::audit_code_via_compiler(&workspace_dir).await {
                                    Ok(_) => { let _ = tx_beta.send(SwarmMessage::CompilerPassed).await; },
                                    Err(e) => { let _ = tx_beta.send(SwarmMessage::CompilerFailed { reason: e.to_string() }).await; }
                                }
                                let _ = crate::swarm::tools::TitaniumGate::destroy_sandbox().await;
                            },
                            Err(e) => {
                                let _ = tx_beta.send(SwarmMessage::CompilerFailed { reason: format!("Lỗi tạo Lồng ấp: {}", e) }).await;
                            }
                        }
                    });

                    // --- KÍCH HOẠT GAMMA (THE AUDITOR) ---
                    let tx_gamma = tx.clone();
                    let code_for_gamma = code;
                    tokio::spawn(async move {
                        debug!("🦅 [Agent Gamma] Quét rác và lỗi bảo mật vật lý...");
                        if code_for_gamma.contains("unwrap()") {
                            let _ = tx_gamma.send(SwarmMessage::SecurityViolated { reason: "Vi phạm kỷ luật Zero-Panic (unwrap detected)".into() }).await;
                        } else {
                            let _ = tx_gamma.send(SwarmMessage::SecurityApproved).await;
                        }
                    });

                    // 🚨 Đóng kết nối gốc để chống Deadlock
                    drop(tx.clone()); 
                }

                SwarmMessage::CompilerPassed => {
                    info!("🛡️ [Agent Beta] Trình biên dịch chấp thuận. Logic chuẩn.");
                    consensus_count += 1;
                }
                SwarmMessage::CompilerFailed { reason } => {
                    warn!("❌ [Agent Beta] Phủ quyết: {}", reason);
                    return Err(OuroborosError::Swarm(format!("Beta Reject: {}", reason)));
                }

                SwarmMessage::SecurityApproved => {
                    info!("🦅 [Agent Gamma] Kiểm định bảo mật: Vô trùng.");
                    consensus_count += 1;
                }
                SwarmMessage::SecurityViolated { reason } => {
                    warn!("❌ [Agent Gamma] Phủ quyết: {}", reason);
                    return Err(OuroborosError::Swarm(format!("Gamma Reject: {}", reason)));
                }
            }

            if consensus_count == 2 { break; }
        }

        // --- BƯỚC 3: THỰC THI VẬT LÝ ---
        info!("⚖️ [Titanium Gate] Đạt 100% đồng thuận. Bắt đầu Hợp nhất nguyên tử.");
        let tools = AgentTools::new("."); 
        tools.apply_mutation(&contract)?;

        Ok(format!("✅ Đột biến thành công file [{}]. Swarm giải tán.", contract.target_file))
    }

    fn extract_and_parse_contract(&self, raw_output: &str) -> Result<FileMutationContract> {
        let json_start = raw_output.find('{');
        let json_end = raw_output.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let clean_json = &raw_output[start..=end];
            serde_json::from_str::<FileMutationContract>(clean_json)
                .map_err(|e| OuroborosError::Swarm(format!("Lỗi Parse Hợp đồng: {}", e)))
        } else {
            Err(OuroborosError::Swarm("Hợp đồng JSON không tồn tại trong luồng suy luận.".into()))
        }
    }
}