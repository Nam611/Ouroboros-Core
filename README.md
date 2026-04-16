# 🌀 Ouroboros Core - Agentic OS
> **"Substance over Scale. Frontier Intelligence over Brute Force."**

Ouroboros Core không phải là một "Thin Wrapper" gọi API qua mạng. Đây là một **Hệ Điều Hành Tác Tử (Agentic OS)** được rèn đúc bằng Rust, chạy trực tiếp trên Bare-Metal với kỷ luật First Principles.

## 🏛️ Kiến Trúc Titanium (The 4 Pillars)
1. **Lõi Nhận Thức (Candle Bridge):** Tích hợp trực tiếp Qwen-0.5B vào VRAM/RAM. Sinh chữ tự hồi quy (Auto-Regressive) với Zero-Allocation Memory.
2. **Lò Phản Ứng Lượng Tử (TurboQuant):** Thuật toán nén Vector bằng SIMD/AVX2, ép xung KV Cache (Float32 -> Int8) với tốc độ Gigabyte/s.
3. **Tàng Kinh Các (Semantic RAG):** Động cơ nhúng (Embedding Engine) all-MiniLM-L6-v2 kết hợp đồ thị tri thức cục bộ, tiêu diệt hoàn toàn ảo giác (Hallucination).
4. **Thống Trị Đa Tác Tử (Swarm Sandbox):** Cách ly Agent bằng Git Worktree. Không một dòng code rác nào được lọt vào Codebase nếu chưa qua kiểm duyệt của Compiler.

## 🚀 Kích hoạt Động Cơ
```bash
cargo build --release
cargo run --release