use crate::error::{OuroborosError, Result};
use rand::Rng;
use std::f32::consts::PI;
use std::sync::Arc;
use std::alloc::{alloc, dealloc, Layout};
use tracing::info;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =========================================================================
// 🚀 [MEMORY KERNEL]: ALIGNED BUFFER (32-BYTE)
// Ép hệ điều hành cấp phát bộ nhớ căn lề chuẩn xác cho AVX2
// =========================================================================

pub struct AlignedBuffer {
    ptr: *mut f32,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    pub fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), 32)
            .expect("❌ Lỗi Layout: Không thể yêu cầu bộ nhớ 32-byte aligned");
        let ptr = unsafe { alloc(layout) as *mut f32 };
        if ptr.is_null() { panic!("❌ Ouroboros OOM: Hết RAM vật lý!"); }
        
        unsafe { std::ptr::write_bytes(ptr, 0, len); } // Zero-init
        Self { ptr, len, layout }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, self.layout); }
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

// =========================================================================
// 🚀 [PHASE 7]: ZERO-ALLOCATION ALIGNED WORKSPACE
// =========================================================================

pub struct TurboWorkspace {
    pub y: AlignedBuffer,
    pub r: AlignedBuffer,
    pub qjl_vec: AlignedBuffer,
    pub x_tilde_mse: AlignedBuffer,
}

impl TurboWorkspace {
    pub fn new(d: usize) -> Self {
        Self {
            y: AlignedBuffer::new(d),
            r: AlignedBuffer::new(d),
            qjl_vec: AlignedBuffer::new(d),
            x_tilde_mse: AlignedBuffer::new(d),
        }
    }
}

// =========================================================================
// ⚡ LÕI TOÁN HỌC HADAMARD (SUPER-SCALAR AVX2)
// =========================================================================

pub struct TurboMathCore {
    pub d: usize,
    pub sign_pi: Vec<f32>, 
    pub sign_s: Vec<f32>,  
}

impl TurboMathCore {
    pub fn new(d: usize) -> Result<Self> {
        info!("⚙️ [TurboQuant FWHT] Khởi tạo Lõi Hadamard Cộng sinh Cơ học (d={})...", d);
        if !d.is_power_of_two() { return Err(OuroborosError::Memory("Chiều d phải là lũy thừa của 2!".into())); }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (d as f32).sqrt();

        let mut sign_pi = Vec::with_capacity(d);
        let mut sign_s = Vec::with_capacity(d);

        for _ in 0..d {
            let s1 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            let s2 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            sign_pi.push(s1 * scale);
            sign_s.push(s2);
        }

        info!("✅ [TurboQuant FWHT] Khởi tạo hoàn tất. Đã ép khuôn bộ nhớ 32-byte!");
        Ok(Self { d, sign_pi, sign_s })
    }

    pub fn get_optimal_centroids(bit_width: usize, d: usize) -> Result<Vec<f32>> {
        let scale = 1.0 / (d as f32).sqrt();
        match bit_width {
            1 => Ok(vec![-(2.0 / PI).sqrt() * scale, (2.0 / PI).sqrt() * scale]),
            2 => Ok(vec![-1.510 * scale, -0.453 * scale, 0.453 * scale, 1.510 * scale]),
            3 => Ok(vec![-2.152 * scale, -1.344 * scale, -0.756 * scale, -0.245 * scale, 0.245 * scale,  0.756 * scale,  1.344 * scale,  2.152 * scale]),
            _ => Err(OuroborosError::Memory("Bit-width chưa hỗ trợ!".into())),
        }
    }

    #[inline(always)]
    pub fn fast_hadamard_transform(data: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::fwht_avx2_unrolled(data); return; }
        }
        
        let n = data.len();
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = data[j];
                    let y = data[j + h];
                    data[j] = x + y;
                    data[j + h] = x - y;
                }
            }
            h *= 2;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fwht_avx2_unrolled(data: &mut [f32]) {
        let n = data.len();
        let mut h = 1;
        let ptr = data.as_mut_ptr();
        
        while h < 8 && h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = *ptr.add(j);
                    let y = *ptr.add(j + h);
                    *ptr.add(j) = x + y;
                    *ptr.add(j + h) = x - y;
                }
            }
            h *= 2;
        }

        // 🚨 KỸ THUẬT UNROLLING (Mở cuộn 4x) & ALIGNED LOAD (Đọc chuẩn 32-byte) & PREFETCH
        while h < n {
            for i in (0..n).step_by(h * 2) {
                let mut j = i;
                while j + 31 < i + h {
                    // Tiên tri: Kéo dữ liệu của vòng lặp tiếp theo vào L1 Cache
                    _mm_prefetch(ptr.add(j + 64) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(ptr.add(j + h + 64) as *const i8, _MM_HINT_T0);

                    // Sử dụng _load_ps (Aligned Load) cực nhanh thay vì _loadu_ps
                    let x0 = _mm256_load_ps(ptr.add(j));
                    let y0 = _mm256_load_ps(ptr.add(j + h));
                    let x1 = _mm256_load_ps(ptr.add(j + 8));
                    let y1 = _mm256_load_ps(ptr.add(j + h + 8));
                    let x2 = _mm256_load_ps(ptr.add(j + 16));
                    let y2 = _mm256_load_ps(ptr.add(j + h + 16));
                    let x3 = _mm256_load_ps(ptr.add(j + 24));
                    let y3 = _mm256_load_ps(ptr.add(j + h + 24));
                    
                    // Superscalar Execution: CPU tính toán độc lập 4 cụm cùng lúc
                    _mm256_store_ps(ptr.add(j), _mm256_add_ps(x0, y0));
                    _mm256_store_ps(ptr.add(j + h), _mm256_sub_ps(x0, y0));
                    _mm256_store_ps(ptr.add(j + 8), _mm256_add_ps(x1, y1));
                    _mm256_store_ps(ptr.add(j + h + 8), _mm256_sub_ps(x1, y1));
                    _mm256_store_ps(ptr.add(j + 16), _mm256_add_ps(x2, y2));
                    _mm256_store_ps(ptr.add(j + h + 16), _mm256_sub_ps(x2, y2));
                    _mm256_store_ps(ptr.add(j + 24), _mm256_add_ps(x3, y3));
                    _mm256_store_ps(ptr.add(j + h + 24), _mm256_sub_ps(x3, y3));
                    
                    j += 32;
                }
                // Xử lý phần lẻ (nếu có)
                while j < i + h {
                    let vec_x = _mm256_loadu_ps(ptr.add(j));
                    let vec_y = _mm256_loadu_ps(ptr.add(j + h));
                    _mm256_storeu_ps(ptr.add(j), _mm256_add_ps(vec_x, vec_y));
                    _mm256_storeu_ps(ptr.add(j + h), _mm256_sub_ps(vec_x, vec_y));
                    j += 8;
                }
            }
            h *= 2;
        }
    }

    #[inline(always)]
    pub fn apply_signs(data: &mut [f32], signs: &[f32]) {
        let d = data.len();
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                let ptr_d = data.as_mut_ptr();
                let ptr_s = signs.as_ptr();
                while i + 31 < d {
                    _mm256_store_ps(ptr_d.add(i), _mm256_mul_ps(_mm256_load_ps(ptr_d.add(i)), _mm256_loadu_ps(ptr_s.add(i))));
                    _mm256_store_ps(ptr_d.add(i + 8), _mm256_mul_ps(_mm256_load_ps(ptr_d.add(i + 8)), _mm256_loadu_ps(ptr_s.add(i + 8))));
                    _mm256_store_ps(ptr_d.add(i + 16), _mm256_mul_ps(_mm256_load_ps(ptr_d.add(i + 16)), _mm256_loadu_ps(ptr_s.add(i + 16))));
                    _mm256_store_ps(ptr_d.add(i + 24), _mm256_mul_ps(_mm256_load_ps(ptr_d.add(i + 24)), _mm256_loadu_ps(ptr_s.add(i + 24))));
                    i += 32;
                }
                while i < d { data[i] *= signs[i]; i += 1; }
                return;
            }
        }
        for i in 0..d { data[i] *= signs[i]; }
    }
}

// =========================================================================
// 🚀 [ALGORITHMS]: MSE & PROD (Giữ nguyên logic Phase 6, cập nhật as_mut_slice)
// =========================================================================

#[derive(Debug)]
pub struct QuantizedMseCache {
    pub indices: Vec<u8>, 
    pub original_len: usize,
}

pub struct TurboQuantMse {
    math_core: Arc<TurboMathCore>,
    centroids: Vec<f32>,
}

impl TurboQuantMse {
    pub fn new(math_core: Arc<TurboMathCore>, bit_width: usize) -> Result<Self> {
        let centroids = TurboMathCore::get_optimal_centroids(bit_width, math_core.d)?;
        Ok(Self { math_core, centroids }) 
    }

    pub fn compress(&self, x: &[f32], ws: &mut TurboWorkspace) -> Result<QuantizedMseCache> {
        let d = self.math_core.d;
        let y_slice = ws.y.as_mut_slice();
        y_slice.copy_from_slice(x);
        
        TurboMathCore::apply_signs(y_slice, &self.math_core.sign_pi);
        TurboMathCore::fast_hadamard_transform(y_slice);

        let mut indices = Vec::with_capacity(d);
        for &val in y_slice.iter() {
            let mut best_idx = 0;
            let mut min_dist = f32::MAX;
            for (i, &c) in self.centroids.iter().enumerate() {
                let dist = (val - c).abs();
                if dist < min_dist { min_dist = dist; best_idx = i as u8; }
            }
            indices.push(best_idx);
        }
        Ok(QuantizedMseCache { indices, original_len: d })
    }

    pub fn decompress_into(&self, cache: &QuantizedMseCache, y_buffer: &mut [f32], out: &mut [f32]) -> Result<()> {
        let d = self.math_core.d;
        for i in 0..d { y_buffer[i] = self.centroids[cache.indices[i] as usize]; }
        
        TurboMathCore::fast_hadamard_transform(y_buffer);
        TurboMathCore::apply_signs(y_buffer, &self.math_core.sign_pi);
        out.copy_from_slice(y_buffer);
        Ok(())
    }
}

#[derive(Debug)]
pub struct QuantizedProdCache {
    pub mse_cache: QuantizedMseCache,
    pub qjl_signs: Vec<i8>, 
    pub residual_norm: f32, 
}

pub struct TurboQuantProd {
    math_core: Arc<TurboMathCore>,
    mse_quantizer: TurboQuantMse,
}

impl TurboQuantProd {
    pub fn new(math_core: Arc<TurboMathCore>, target_bit_width: usize) -> Result<Self> {
        let mse_quantizer = TurboQuantMse::new(Arc::clone(&math_core), target_bit_width - 1)?;
        Ok(Self { math_core, mse_quantizer })
    }

    pub fn compress(&self, x: &[f32], ws: &mut TurboWorkspace) -> Result<QuantizedProdCache> {
        let d = self.math_core.d;
        let mse_cache = self.mse_quantizer.compress(x, ws)?;
        
        self.mse_quantizer.decompress_into(&mse_cache, ws.y.as_mut_slice(), ws.x_tilde_mse.as_mut_slice())?;

        let mut r_norm_sq = 0.0;
        let r_slice = ws.r.as_mut_slice();
        let x_tilde_slice = ws.x_tilde_mse.as_slice();
        
        for i in 0..d {
            let diff = x[i] - x_tilde_slice[i];
            r_slice[i] = diff;
            r_norm_sq += diff * diff;
        }
        let residual_norm = r_norm_sq.sqrt();

        let qjl_vec_slice = ws.qjl_vec.as_mut_slice();
        qjl_vec_slice.copy_from_slice(r_slice);
        TurboMathCore::apply_signs(qjl_vec_slice, &self.math_core.sign_s);
        TurboMathCore::fast_hadamard_transform(qjl_vec_slice);
        
        let mut qjl_signs = Vec::with_capacity(d);
        for &val in qjl_vec_slice.iter() {
            qjl_signs.push(if val >= 0.0 { 1 } else { -1 });
        }

        Ok(QuantizedProdCache { mse_cache, qjl_signs, residual_norm })
    }

    pub fn decompress_into(&self, cache: &QuantizedProdCache, ws: &mut TurboWorkspace, out: &mut [f32]) -> Result<()> {
        let d = self.math_core.d;
        self.mse_quantizer.decompress_into(&cache.mse_cache, ws.y.as_mut_slice(), ws.x_tilde_mse.as_mut_slice())?;
        
        let qjl_scale = (std::f32::consts::PI / 2.0).sqrt() / (d as f32) * cache.residual_norm;
        let qjl_vec_slice = ws.qjl_vec.as_mut_slice();
        
        for i in 0..d {
            qjl_vec_slice[i] = cache.qjl_signs[i] as f32;
        }
        
        TurboMathCore::fast_hadamard_transform(qjl_vec_slice);
        TurboMathCore::apply_signs(qjl_vec_slice, &self.math_core.sign_s);

        let x_tilde_slice = ws.x_tilde_mse.as_slice();
        for i in 0..d {
            out[i] = x_tilde_slice[i] + (qjl_scale * qjl_vec_slice[i]);
        }
        Ok(())
    }
}