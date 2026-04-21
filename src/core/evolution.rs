use crate::error::{OuroborosError, Result};
use crate::core::query_loop::{AgentTask, QueryLoop};
use crate::reasoning::candle_bridge::CandleBridge;
use memmap2::MmapOptions;
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn, error};
use tokio::time::{sleep, Duration};

/// 🌌 [THE SINGULARITY ENGINE]
/// Động cơ Tiến hóa Tự trị. Quét, phân tích và tự đột biến mã nguồn bằng Zero-Copy I/O.
pub struct EvolutionDaemon {
    project_root: PathBuf,
    scan_interval: Duration,
    brain: CandleBridge, 
}

impl EvolutionDaemon {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            project_root: root.as_ref().to_path_buf(),
            scan_interval: Duration::from_secs(600),
            brain: CandleBridge::new(), 
        }
    }

    /// 🔥 NÂNG CẤP: &mut self để cho phép não bộ Gemma cập nhật KV Cache
    pub async fn ignite(&mut self, ouroboros: &mut QueryLoop) -> Result<()> {
        info!("🌌 [THE SINGULARITY] Ouroboros Evolution Engine has been ignited.");
        info!("🛡️ Titanium Protocol is active. Starting self-reflection loop...");

        if let Err(e) = self.brain.boot_sequence() {
            error!("❌ Lỗi khởi động Mạng Nơ-ron: {:?}", e);
            return Err(e);
        }

        loop {
            // Xử lý lỗi trích xuất DNA để Daemon không bị sập
            let source_files = match self.gather_dna(&self.project_root) {
                Ok(files) => files,
                Err(e) => {
                    error!("❌ Lỗi trích xuất DNA mã nguồn: {:?}", e);
                    sleep(self.scan_interval).await;
                    continue;
                }
            };

            if source_files.is_empty() {
                warn!("⚠️ No source code found for evolution.");
                sleep(self.scan_interval).await;
                continue;
            }

            for file_path in source_files {
                let mut mutation_task: Option<AgentTask> = None;

                {
                    // Cơ chế sinh tồn của Daemon: Lỗi đọc 1 file thì bỏ qua, không sập toàn hệ thống
                    let file = match File::open(&file_path) {
                        Ok(f) => f,
                        Err(e) => {
                            error!("❌ Failed to open file for mmap: {:?}. Error: {}", file_path, e);
                            continue;
                        }
                    };

                    let mmap = unsafe {
                        match MmapOptions::new().map(&file) {
                            Ok(m) => m,
                            Err(e) => {
                                error!("❌ Mmap mapping failed for {:?}: {}", file_path, e);
                                continue;
                            }
                        }
                    };

                    if let Ok(content_view) = std::str::from_utf8(&mmap) {
                        debug!("⚡ Zero-copy view established for {:?}", file_path);
                        
                        if let Some(task) = self.analyze_with_gemma(content_view, &file_path) {
                            mutation_task = Some(task);
                        }
                    }
                } // Giải phóng file mmap ra khỏi RAM

                if let Some(task) = mutation_task {
                    info!("🎯 Evolutionary bottleneck detected in {:?}. Triggering Swarm Mutation...", file_path);
                    
                    if let Err(e) = ouroboros.run(task).await {
                        error!("❌ Self-evolution failed for {:?}: {:?}", file_path, e);
                    } else {
                        info!("✅ [MUTATION SUCCESS] File {:?}. Codebase has evolved.", file_path);
                    }
                }
            }

            info!("💤 Evolution cycle completed. System cooling down...");
            sleep(self.scan_interval).await;
        }
    }

    fn gather_dna(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        if dir.is_dir() {
            // 🚨 TITANIUM FIX: Khóa chặt kiểu dữ liệu std::io::Error để triệt tiêu lỗi E0282
            for entry in std::fs::read_dir(dir).map_err(|e: std::io::Error| OuroborosError::System(e.to_string()))? {
                let entry = entry.map_err(|e: std::io::Error| OuroborosError::System(e.to_string()))?;
                let path = entry.path();
                let path_str = path.to_string_lossy();
                
                if path_str.contains("target") || path_str.contains(".ouroboros_sandbox") || path_str.contains(".git") {
                    continue;
                }

                if path.is_dir() {
                    files.extend(self.gather_dna(&path)?);
                } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    files.push(path);
                }
            }
        }
        Ok(files)
    }

    /// 🔥 NÂNG CẤP: &mut self để cập nhật KV Cache của Não bộ
    fn analyze_with_gemma(&mut self, code_content: &str, file_path: &Path) -> Option<AgentTask> {
        let path_str = file_path.to_string_lossy();
        
        match self.brain.analyze_source_code(code_content, &path_str) {
            Ok(Some(task)) => Some(task),
            Ok(None) => None,
            Err(e) => {
                error!("❌ Gemma Reasoning failed on {:?}: {:?}", file_path, e);
                None
            }
        }
    }
}