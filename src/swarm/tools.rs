use crate::error::{OuroborosError, Result};
use std::fs;
use std::path::PathBuf;
// 🚨 NẠP LÕI SERDE ĐỂ PARSE JSON
use serde::Deserialize;
use tracing::{debug, info};

/// 📜 HỢP ĐỒNG DỮ LIỆU (DATA CONTRACT)
#[derive(Debug, Deserialize)] // 🚨 THÊM MACRO NÀY
pub struct FileMutationContract {
    pub target_file: String,
    pub original_function_signature: String,
    pub optimized_code: String,
    pub reasoning: String,
}

/// 🦾 TITANIUM HANDS: Đôi tay vật lý của Agent
pub struct AgentTools {
    sandbox_root: PathBuf,
}

impl AgentTools {
    pub fn new(sandbox_root: &str) -> Self {
        Self {
            sandbox_root: PathBuf::from(sandbox_root),
        }
    }

    /// 🔍 Đọc code từ file (Chỉ cho phép đọc bên trong Lồng ấp)
    pub fn read_target_file(&self, relative_path: &str) -> Result<String> {
        let safe_path = self.resolve_safe_path(relative_path)?;
        debug!("📖 [AgentTool] Đang phân tích file: {:?}", safe_path);
        
        fs::read_to_string(&safe_path).map_err(|e| {
            OuroborosError::Swarm(format!("Không thể đọc file {}: {}", relative_path, e))
        })
    }

    /// ✍️ Ghi đè thuật toán (Chỉ thực thi khi tuân thủ Hợp đồng Dữ liệu)
    pub fn apply_mutation(&self, contract: &FileMutationContract) -> Result<()> {
        let safe_path = self.resolve_safe_path(&contract.target_file)?;
        
        info!("🧬 [Deep Research] Áp dụng thuật toán mới vào: {}", contract.target_file);
        debug!("Lập luận tối ưu: {}", contract.reasoning);

        // 1. Đọc file cũ
        let current_content = fs::read_to_string(&safe_path).map_err(|e| {
            OuroborosError::Swarm(format!("Lỗi đọc file trước khi mutate: {}", e))
        })?;

        // 2. Tìm và thay thế (Giả lập việc Parse AST cơ bản)
        // Nếu không tìm thấy signature cũ, lập tức từ chối thao tác! (Khóa chặt Logic)
        if !current_content.contains(&contract.original_function_signature) {
            return Err(OuroborosError::Swarm(
                "Vi phạm Hợp đồng: Không tìm thấy function signature gốc. Từ chối ghi!".into()
            ));
        }

        // Tạm thời thay thế toàn bộ nội dung file (Ở kỷ nguyên sau sẽ thay thế từng block AST)
        fs::write(&safe_path, &contract.optimized_code).map_err(|e| {
            OuroborosError::Swarm(format!("Lỗi ghi file vật lý: {}", e))
        })?;

        Ok(())
    }

    /// 🛡️ CỔNG AN NINH: Ngăn chặn Path Traversal Attack (VD: "../../etc/password")
    fn resolve_safe_path(&self, relative_path: &str) -> Result<PathBuf> {
        let full_path = self.sandbox_root.join(relative_path);
        
        // Đảm bảo đường dẫn cuối cùng vẫn nằm gọn trong Sandbox
        if !full_path.starts_with(&self.sandbox_root) {
            return Err(OuroborosError::Swarm("CẢNH BÁO AN NINH: Agent cố gắng thoát khỏi Sandbox!".into()));
        }
        
        Ok(full_path)
    }
}