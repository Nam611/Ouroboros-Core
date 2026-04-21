use crate::error::{OuroborosError, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::process::Command;
use serde::Deserialize;
use tracing::{debug, info, warn};

/// 📜 HỢP ĐỒNG DỮ LIỆU (DATA CONTRACT)
#[derive(Debug, Deserialize, Clone)]
pub struct FileMutationContract {
    pub target_file: String,
    pub original_function_signature: String,
    pub optimized_code: String,
    pub reasoning: String,
}

/// 🦾 TITANIUM HANDS: Đôi tay vật lý của Agent Alpha (Mutate File)
pub struct AgentTools {
    sandbox_root: PathBuf,
}

impl AgentTools {
    pub fn new(sandbox_root: &str) -> Self {
        Self {
            sandbox_root: PathBuf::from(sandbox_root),
        }
    }

    pub fn read_target_file(&self, relative_path: &str) -> Result<String> {
        let safe_path = self.resolve_safe_path(relative_path)?;
        debug!("📖 [AgentTool] Đang phân tích file: {:?}", safe_path);
        fs::read_to_string(&safe_path).map_err(|e| {
            OuroborosError::Swarm(format!("Không thể đọc file {}: {}", relative_path, e))
        })
    }

    pub fn apply_mutation(&self, contract: &FileMutationContract) -> Result<()> {
        let safe_path = self.resolve_safe_path(&contract.target_file)?;
        info!("🧬 [Deep Research] Áp dụng thuật toán mới vào: {}", contract.target_file);
        
        let current_content = fs::read_to_string(&safe_path).map_err(|e| {
            OuroborosError::Swarm(format!("Lỗi đọc file trước khi mutate: {}", e))
        })?;

        if !current_content.contains(&contract.original_function_signature) {
            return Err(OuroborosError::Swarm(
                "Vi phạm Hợp đồng: Không tìm thấy function signature gốc. Từ chối ghi!".into()
            ));
        }

        fs::write(&safe_path, &contract.optimized_code).map_err(|e| {
            OuroborosError::Swarm(format!("Lỗi ghi file vật lý: {}", e))
        })?;

        Ok(())
    }

    fn resolve_safe_path(&self, relative_path: &str) -> Result<PathBuf> {
        let full_path = self.sandbox_root.join(relative_path);
        if !full_path.starts_with(&self.sandbox_root) {
            return Err(OuroborosError::Swarm("CẢNH BÁO AN NINH: Agent cố gắng thoát khỏi Sandbox!".into()));
        }
        Ok(full_path)
    }
}

// =========================================================================
// 🚨 MẢNH GHÉP PHASE 3: TITANIUM GATE (LỚP KHIÊN COMPILER CỦA AGENT BETA)
// =========================================================================

pub struct TitaniumGate;

impl TitaniumGate {
    /// Agent Beta tạo môi trường cách ly (Trả về String path)
    pub async fn create_sandbox(target_file_path: &str, new_code: &str) -> Result<String> {
        let sandbox_dir = ".ouroboros_sandbox/temp_run";
        
        if !Path::new(sandbox_dir).exists() {
            tokio::fs::create_dir_all(sandbox_dir).await.map_err(|e| {
                OuroborosError::System(format!("Không thể đúc lồng ấp: {}", e))
            })?;
        }

        // Tạo cấu trúc thư mục rỗng bên trong sandbox nếu cần (vd: src/main.rs)
        let file_path = Path::new(target_file_path);
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                OuroborosError::System(format!("Lỗi tạo cấu trúc thư mục con: {}", e))
            })?;
        }

        tokio::fs::write(target_file_path, new_code).await.map_err(|e| {
            OuroborosError::System(format!("Lỗi ghi file vật lý vào Lồng ấp: {}", e))
        })?;

        // Trả về thư mục gốc (root) để cargo check có thể chạy
        Ok(".".to_string()) 
    }

    /// Gọi Trình biên dịch Rust
    pub async fn audit_code_via_compiler(workspace_dir: &str) -> Result<()> {
        info!("⚖️ [Titanium Gate] Đang gọi Trình biên dịch kiểm tra AST tại: {}", workspace_dir);

        let output = Command::new("cargo")
            .arg("check")
            .current_dir(workspace_dir)
            .output()
            .await
            .map_err(|e: std::io::Error| OuroborosError::Reasoning(e.to_string()))?; // 🚨 KHAI BÁO RÕ std::io::Error

        if output.status.success() {
            info!("✅ [Titanium Gate] Mã nguồn vô trùng. Compiler hoàn toàn chấp thuận!");
            Ok(())
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr).to_string();
            warn!("❌ [Titanium Gate] Compiler từ chối mã nguồn! Trả về lỗi cho Agent Alpha.");
            Err(OuroborosError::System(error_msg))
        }
    }

    /// Thiêu rụi Lồng ấp
    pub async fn destroy_sandbox() -> Result<()> {
        let sandbox_dir = ".ouroboros_sandbox/temp_run";
        if Path::new(sandbox_dir).exists() {
            tokio::fs::remove_dir_all(sandbox_dir).await.map_err(|e| {
                OuroborosError::System(format!("Không thể tiêu hủy lồng ấp: {}", e))
            })?;
            debug!("🔥 [Titanium Gate] Đã thiêu rụi lồng ấp ô nhiễm.");
        }
        Ok(())
    }
}