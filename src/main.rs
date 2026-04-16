pub mod core;
pub mod error;
pub mod memory;
pub mod network;
pub mod reasoning;
pub mod swarm;

use tracing_subscriber::{EnvFilter, FmtSubscriber};

// 🚨 BỎ MACRO #[tokio::main] - CHÚNG TA GIÀNH LẠI QUYỀN KIỂM SOÁT VẬT LÝ TỐI ĐA
fn main() -> anyhow::Result<()> {
    // 1. TỊNH HÓA OS NGAY TẠI MILI-GIÂY ĐẦU TIÊN (Khởi động lạnh)
    // Lúc này chưa có bất kỳ một luồng Tokio nào được sinh ra. OS hoàn toàn bất lực.
    let toxic_vars = [
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", 
        "all_proxy", "ALL_PROXY", "HF_ENDPOINT"
    ];
    for var in toxic_vars {
        std::env::remove_var(var);
    }

    // Khóa chặt Endpoint chuẩn xác và cấm tuyệt đối Proxy
    std::env::set_var("HF_ENDPOINT", "https://huggingface.co");
    std::env::set_var("NO_PROXY", "*");

    // 2. Khởi động hệ thống viễn trắc
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::DEBUG.into()))
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    tracing::info!("🛡️ [Titanium Protocol] Khởi động lạnh hoàn tất. Mạng lưới đã được vô trùng tuyệt đối.");

    // 3. TỰ TAY ĐÚC ĐỘNG CƠ BẤT ĐỒNG BỘ (TOKIO RUNTIME)
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // 4. Bơm năng lượng vào Trái tim Ouroboros
    runtime.block_on(async {
        let mut ouroboros = core::query_loop::QueryLoop::new();
        ouroboros.run().await
    })?;

    Ok(())
}