use crate::error::Result;
use tracing::info;

/// BẢN HỢP ĐỒNG GIAO TIẾP MẠNG (XDP/eBPF BRIDGE)
/// Cổng kết nối siêu tốc trực tiếp với Card mạng (NIC) thông qua eBPF.
pub trait XdpEngine: Send + Sync {
    /// Đính kèm chương trình eBPF vào interface mạng (VD: eth0)
    fn attach_to_interface(&mut self, interface: &str) -> Result<()>;
    
    /// Đọc dữ liệu Zero-copy trực tiếp từ eBPF Maps vào RAM của Ouroboros
    fn poll_zero_copy_events(&self) -> Result<()>;
}

/// Mô phỏng XDP Bridge (Sẽ thay bằng `aya-bpf` ở Phase tiếp theo)
pub struct NativeXdpBridge {
    interface: Option<String>,
}

impl NativeXdpBridge {
    pub fn new() -> Self {
        Self { interface: None }
    }
}

impl XdpEngine for NativeXdpBridge {
    fn attach_to_interface(&mut self, interface: &str) -> Result<()> {
        self.interface = Some(interface.to_string());
        info!("🛡️ [XDP Bridge] Đã thiết lập màng lọc eBPF ảo tại interface: {}", interface);
        info!("-> (Phase tới sẽ dùng thư viện `aya` để cấy mã máy vào Kernel Linux)");
        Ok(())
    }

    fn poll_zero_copy_events(&self) -> Result<()> {
        // Thực tế: Lấy dữ liệu từ BPF RingBuf mà không cần copy
        Ok(())
    }
}