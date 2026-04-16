use crate::error::{OuroborosError, Result};
use std::process::Command;
use tracing::{debug, info, warn};

/// 🦂 SWARM COORDINATOR: Hệ thống điều phối Tác tử với Lớp khiên Titanium
pub struct SwarmCoordinator {
    max_agents: usize,
    active_agents: usize,
}

impl SwarmCoordinator {
    pub fn new(max_agents: usize) -> Self {
        info!("🦂 [Swarm] Kích hoạt buồng ấp Đa Tác Tử. Giới hạn: {} Agents đồng thời.", max_agents);
        Self {
            max_agents,
            active_agents: 0,
        }
    }

    /// Điều phối nhiệm vụ cho Agent (Cô lập hoàn toàn với Codebase gốc)
    pub async fn dispatch_task(&mut self, task: &str) -> Result<String> {
        if self.active_agents >= self.max_agents {
            return Err(OuroborosError::Swarm("Quá tải Swarm. Hết slot Agent.".to_string()));
        }

        self.active_agents += 1;
        let agent_id = format!("Agent_Alpha_{:02}", self.active_agents);
        
        info!("🚀 Khởi chạy {} thực thi nhiệm vụ: [{}]", agent_id, task);

        // 🛡️ BƯỚC 1: TẠO KHÔNG GIAN CÁCH LY (Git Worktree Sandbox)
        let sandbox_path = format!(".ouroboros_sandbox/{}", agent_id);
        self.setup_git_worktree(&sandbox_path, &agent_id)?;

        // 🛠️ BƯỚC 2: MÔ PHỎNG AGENT THỰC THI (Ở Kỷ nguyên sau sẽ nối với LoopLM)
        debug!("⚙️ [{}] Đang thao tác trong lồng ấp: {}...", agent_id, sandbox_path);
        // Tạm thời giả lập việc Agent đang sửa file Cargo.toml trong Sandbox
        std::thread::sleep(std::time::Duration::from_millis(500));

        // ⚖️ BƯỚC 3: TITANIUM GATE - KIỂM DUYỆT MÃ NGUỒN
        // Dùng Compiler của Rust làm giám khảo chấm điểm code của Agent
        let verify_status = self.verify_sandbox_code(&sandbox_path);

        // 🧹 BƯỚC 4: DỌN DẸP KHÔNG GIAN
        self.cleanup_git_worktree(&sandbox_path, &agent_id)?;
        self.active_agents -= 1;

        match verify_status {
            Ok(_) => {
                info!("✅ [Titanium Gate] {} hoàn thành xuất sắc. Code sạch!", agent_id);
                Ok(format!("Hoàn tất vật lý: {} (Đã vượt qua kiểm duyệt)", task))
            },
            Err(e) => {
                warn!("❌ [Titanium Gate] Cảnh báo: {} viết code lỗi! Đã tiêu hủy lồng ấp.", agent_id);
                Err(e)
            }
        }
    }

    /// 🛡️ Lõi Cách ly: Tạo Git Worktree (Nhánh tạm thời)
    fn setup_git_worktree(&self, path: &str, branch: &str) -> Result<()> {
        debug!("🛡️ [Sandbox] Cấp phát không gian cách ly (Git Worktree) tại: {}", path);
        
        // Bỏ qua lỗi nếu branch đã tồn tại từ lần sập nguồn trước
        let _ = Command::new("git").args(["worktree", "remove", "-f", path]).output();
        let _ = Command::new("git").args(["branch", "-D", branch]).output();

        let status = Command::new("git")
            .args(["worktree", "add", "-b", branch, path])
            .status()
            .map_err(|e| OuroborosError::Swarm(format!("Lỗi gọi Git: {}", e)))?;

        if !status.success() {
            // Nếu dự án chưa khởi tạo Git, ta báo lỗi nhắc nhở
            return Err(OuroborosError::Swarm("Không thể tạo Worktree. (Gợi ý: Chạy `git init` ở thư mục gốc trước)".to_string()));
        }
        Ok(())
    }

    /// ⚖️ Lõi Kiểm duyệt: Ép Compiler kiểm tra code của Agent
    fn verify_sandbox_code(&self, path: &str) -> Result<()> {
        debug!("⚖️ [Titanium Gate] Đang gọi Trình biên dịch kiểm tra lồng ấp...");
        let status = Command::new("cargo")
            .args(["check", "--manifest-path", &format!("{}/Cargo.toml", path)])
            .status();

        match status {
            Ok(s) if s.success() => Ok(()),
            _ => Err(OuroborosError::Swarm("Agent phá hỏng codebase. Bị từ chối!".to_string()))
        }
    }

    /// 🧹 Lõi Dọn dẹp: Hủy Worktree
    fn cleanup_git_worktree(&self, path: &str, branch: &str) -> Result<()> {
        debug!("🧹 [Sandbox] Tiêu hủy không gian cách ly...");
        let _ = Command::new("git").args(["worktree", "remove", "-f", path]).status();
        let _ = Command::new("git").args(["branch", "-D", branch]).status();
        Ok(())
    }
}