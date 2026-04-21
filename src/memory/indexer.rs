use crate::error::Result;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// 🚨 PROJECT SCANNER: Con mắt quét mã nguồn bằng thuật toán DFS (Zero-Bloat)
pub struct ProjectScanner {
    // Danh sách các \"vùng cấm\" không được phép quét để tiết kiệm CPU và I/O
    ignored_dirs: Vec<String>,
}

impl ProjectScanner {
    pub fn new() -> Self {
        Self {
            ignored_dirs: vec![
                "target".to_string(),
                ".git".to_string(),
                ".ouroboros_sandbox".to_string(),
                "node_modules".to_string(),
                "debug".to_string(),
                "release".to_string(),
            ],
        }
    }

    /// Thuật toán Iterative Depth-First Search (Không dùng đệ quy, chống Stack Overflow)
    pub async fn scan_directory(&self, root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        let root_path = root.as_ref();
        if !root_path.exists() {
            return Err(crate::error::OuroborosError::Reasoning(
                format!("Thư mục không tồn tại: {:?}", root_path)
            ));
        }

        let mut rust_files = Vec::new();
        // Sử dụng Vec như một Stack để lưu trữ các thư mục cần duyệt
        let mut stack = vec![root_path.to_path_buf()];

        info!("👁️ Kích hoạt Radar Quét dự án tại: {:?}", root_path);

        // Vòng lặp I/O không chặn (Non-blocking I/O loop)
        while let Some(current_dir) = stack.pop() {
            let mut entries = match fs::read_dir(&current_dir).await {
                Ok(e) => e,
                Err(e) => {
                    warn!("⚠️ Không quyền truy cập hoặc lỗi đọc thư mục {:?}: {}", current_dir, e);
                    continue;
                }
            };

            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();

                if path.is_dir() {
                    // Nếu là thư mục, kiểm tra xem có nằm trong danh sách đen không
                    if !self.ignored_dirs.contains(&file_name) {
                        stack.push(path); // Đẩy vào Stack để quét tiếp
                    } else {
                        debug!("🚧 Bỏ qua vùng cấm: {:?}", path);
                    }
                } else if path.is_file() {
                    // Nếu là file, kiểm tra đuôi mở rộng có phải là `.rs` (Rust) không
                    if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                        rust_files.push(path);
                    }
                }
            }
        }

        info!("✅ Quét hoàn tất. Phát hiện {} tệp tin mã nguồn Rust (.rs) hợp lệ.", rust_files.len());
        Ok(rust_files)
    }

    /// Đọc nội dung file an toàn với bộ nhớ
    pub async fn read_file_content(&self, path: &Path) -> Result<String> {
        let content = fs::read_to_string(path).await.map_err(|e| {
            crate::error::OuroborosError::Reasoning(format!("Lỗi đọc file {:?}: {}", path, e))
        })?;
        Ok(content)
    }
}