//! Weight-only quantization: int8 and int4 with one scale per group of 32
//! input positions per output row (symmetric, round to nearest, with the
//! clipping range chosen to minimise error), and the matrix-multiply
//! kernels that read those weights.
//!
//! Weights are packed in the same 32-row panels as bf16 matrices and
//! widened to f32 inside registers, so a decode step streams 1.1 (int8) or
//! 0.6 (int4) bytes per weight instead of 2. Each output element is still
//! one fixed sequence of operations: for every group in order, a chain of
//! fused multiply-adds over its 32 positions from zero, scaled into the
//! running sum with one more fused multiply-add. Nothing depends on the
//! batch, the tiling or the instruction set, so the portable and SIMD paths
//! agree bit for bit, as for bf16.

use crate::kernels::{Isa, PANEL, isa, tile_shape};
use crate::pool::Out;

/// Input positions per scale.
pub const GROUP: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Bf16,
    Int8,
    Int4,
}

impl Quant {
    pub fn parse(s: &str) -> Option<Quant> {
        match s {
            "bf16" | "none" => Some(Quant::Bf16),
            "int8" | "q8" => Some(Quant::Int8),
            "int4" | "q4" => Some(Quant::Int4),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Quant::Bf16 => "bf16",
            Quant::Int8 => "int8",
            Quant::Int4 => "int4",
        }
    }

    fn range(self) -> (i32, i32) {
        match self {
            Quant::Int8 => (-127, 127),
            Quant::Int4 => (-8, 7),
            Quant::Bf16 => unreachable!(),
        }
    }

    /// Bytes per input position per panel.
    fn bytes_per_k(self) -> usize {
        match self {
            Quant::Int8 => PANEL,
            Quant::Int4 => PANEL / 2,
            Quant::Bf16 => unreachable!(),
        }
    }
}

/// A quantized `rows × cols` matrix, packed in panels of 32 rows.
#[derive(Clone)]
pub struct QMatrix {
    pub kind: Quant,
    pub rows: usize,
    pub cols: usize,
    /// Panel p, position k: `bytes_per_k` bytes. int8: one signed byte per
    /// row. int4: byte i holds row i (low nibble) and row i + 16 (high
    /// nibble), each offset by 8.
    q: Vec<u8>,
    /// Panel p, group g: 32 scales, one per row.
    s: Vec<f32>,
}

impl QMatrix {
    /// Quantizes row-major f32 weights. `importance`, if given, weights the
    /// error of each input position when choosing a group's clipping range
    /// (larger for positions whose activations are larger).
    pub fn quantize(kind: Quant, rows: usize, cols: usize, w: &[f32], importance: Option<&[f32]>) -> QMatrix {
        assert!(kind != Quant::Bf16);
        assert_eq!(w.len(), rows * cols);
        assert_eq!(rows % PANEL, 0);
        assert_eq!(cols % GROUP, 0, "input dimension must be a multiple of {GROUP}");
        let groups = cols / GROUP;
        let bpk = kind.bytes_per_k();
        let mut q = vec![0u8; rows / PANEL * cols * bpk];
        let mut s = vec![0f32; rows / PANEL * groups * PANEL];
        let (lo, hi) = kind.range();
        let clips: &[f32] = match kind {
            Quant::Int8 => &[1.0, 0.98, 0.96, 0.94, 0.92, 0.9],
            _ => &[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
        };
        let mut codes = [0i32; GROUP];
        let mut best_codes = [0i32; GROUP];
        for r in 0..rows {
            let (p, c) = (r / PANEL, r % PANEL);
            for g in 0..groups {
                let vals = &w[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
                let imp = importance.map(|i| &i[g * GROUP..(g + 1) * GROUP]);
                let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
                let (mut best_err, mut best_scale) = (f64::INFINITY, 0f32);
                if amax > 0.0 {
                    for &clip in clips {
                        let scale = clip * amax / hi as f32;
                        let mut err = 0f64;
                        for (k, &v) in vals.iter().enumerate() {
                            let code = ((v / scale).round() as i32).clamp(lo, hi);
                            codes[k] = code;
                            let d = f64::from(v - code as f32 * scale);
                            err += d * d * imp.map_or(1.0, |i| f64::from(i[k]));
                        }
                        if err < best_err {
                            best_err = err;
                            best_scale = scale;
                            best_codes.copy_from_slice(&codes);
                        }
                    }
                } else {
                    best_codes.fill(0);
                }
                s[(p * groups + g) * PANEL + c] = best_scale;
                for (kk, &code) in best_codes.iter().enumerate() {
                    let k = g * GROUP + kk;
                    let base = (p * cols + k) * bpk;
                    match kind {
                        Quant::Int8 => q[base + c] = code as i8 as u8,
                        _ => {
                            let nib = (code + 8) as u8;
                            if c < 16 {
                                q[base + c] = (q[base + c] & 0xF0) | nib;
                            } else {
                                q[base + c - 16] = (q[base + c - 16] & 0x0F) | (nib << 4);
                            }
                        }
                    }
                }
            }
        }
        QMatrix { kind, rows, cols, q, s }
    }

    /// The integer code of row `r` at position `k`.
    fn code(&self, r: usize, k: usize) -> i32 {
        let (p, c) = (r / PANEL, r % PANEL);
        let base = (p * self.cols + k) * self.kind.bytes_per_k();
        match self.kind {
            Quant::Int8 => i32::from(self.q[base + c] as i8),
            _ => {
                let b = self.q[base + c % 16];
                i32::from(if c < 16 { b & 0x0F } else { b >> 4 }) - 8
            }
        }
    }

    fn scale(&self, r: usize, g: usize) -> f32 {
        let (p, c) = (r / PANEL, r % PANEL);
        self.s[(p * (self.cols / GROUP) + g) * PANEL + c]
    }

    /// Row `r`, dequantized.
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        for (k, o) in out.iter_mut().enumerate().take(self.cols) {
            *o = self.code(r, k) as f32 * self.scale(r, k / GROUP);
        }
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + self.s.len() * 4
    }
}

/// Rows of `x` per cache block, and inner positions per block (a multiple
/// of the group size).
const MC: usize = 60;
const KC: usize = 256;

/// Output panels `p0..p1` of `y = x Wᵀ` for every row of `x`.
pub(crate) fn panels_of(x: &[f32], m: usize, w: &QMatrix, p0: usize, p1: usize, out: Out<f32>) {
    let (n, k) = (w.rows, w.cols);
    let bpk = w.kind.bytes_per_k();
    let groups = k / GROUP;
    for m0 in (0..m).step_by(MC) {
        let m1 = (m0 + MC).min(m);
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            let mut i = m0;
            while i < m1 {
                let (mr, np) = tile_shape(m1 - i);
                let mut p = p0;
                while p < p1 {
                    let np = np.min(p1 - p);
                    let t = QTile {
                        kind: w.kind,
                        x: &x[i * k + k0..],
                        xs: k,
                        q: &w.q[(p * k + k0) * bpk..],
                        qs: k * bpk,
                        s: &w.s[(p * groups + k0 / GROUP) * PANEL..],
                        ss: groups * PANEL,
                        kc,
                        y: out,
                        y0: i * n + p * PANEL,
                        ys: n,
                        init: k0 == 0,
                    };
                    t.run(mr, np);
                    p += np;
                }
                i += mr;
            }
        }
    }
}

pub(crate) struct QTile<'a> {
    kind: Quant,
    x: &'a [f32],
    xs: usize,
    q: &'a [u8],
    qs: usize,
    s: &'a [f32],
    ss: usize,
    kc: usize,
    y: Out<f32>,
    y0: usize,
    ys: usize,
    init: bool,
}

impl QTile<'_> {
    fn run(&self, mr: usize, np: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            macro_rules! dispatch {
                ($m:ident) => {
                    // SAFETY: the tile lies within x, q, s and this task's
                    // columns of y.
                    unsafe {
                        match (mr, np, self.kind) {
                            (1, 1, Quant::Int8) => return $m::tile::<1, 1, false>(self),
                            (1, 2, Quant::Int8) => return $m::tile::<1, 2, false>(self),
                            (1, 3, Quant::Int8) => return $m::tile::<1, 3, false>(self),
                            (1, 4, Quant::Int8) => return $m::tile::<1, 4, false>(self),
                            (2, 1, Quant::Int8) => return $m::tile::<2, 1, false>(self),
                            (2, 2, Quant::Int8) => return $m::tile::<2, 2, false>(self),
                            (3, 1, Quant::Int8) => return $m::tile::<3, 1, false>(self),
                            (4, 1, Quant::Int8) => return $m::tile::<4, 1, false>(self),
                            (4, 2, Quant::Int8) => return $m::tile::<4, 2, false>(self),
                            (8, 1, Quant::Int8) => return $m::tile::<8, 1, false>(self),
                            (1, 1, _) => return $m::tile::<1, 1, true>(self),
                            (1, 2, _) => return $m::tile::<1, 2, true>(self),
                            (1, 3, _) => return $m::tile::<1, 3, true>(self),
                            (1, 4, _) => return $m::tile::<1, 4, true>(self),
                            (2, 1, _) => return $m::tile::<2, 1, true>(self),
                            (2, 2, _) => return $m::tile::<2, 2, true>(self),
                            (3, 1, _) => return $m::tile::<3, 1, true>(self),
                            (4, 1, _) => return $m::tile::<4, 1, true>(self),
                            (4, 2, _) => return $m::tile::<4, 2, true>(self),
                            (8, 1, _) => return $m::tile::<8, 1, true>(self),
                            other => unreachable!("tile {other:?}"),
                        }
                    }
                };
            }
            match isa() {
                Isa::Avx512 => dispatch!(avx512),
                Isa::Avx2 => dispatch!(avx2),
                Isa::Portable => {}
            }
        }
        self.portable(mr, np);
    }

    fn portable(&self, mr: usize, np: usize) {
        let bpk = self.kind.bytes_per_k();
        for i in 0..mr {
            for p in 0..np {
                for c in 0..PANEL {
                    let at = self.y0 + i * self.ys + p * PANEL + c;
                    // SAFETY: this task owns these columns of y.
                    let mut acc = if self.init { 0.0 } else { unsafe { self.y.read(at) } };
                    for g in 0..self.kc / GROUP {
                        let mut gs = 0f32;
                        for kk in g * GROUP..(g + 1) * GROUP {
                            let base = p * self.qs + kk * bpk;
                            let code = match self.kind {
                                Quant::Int8 => i32::from(self.q[base + c] as i8),
                                _ => {
                                    let b = self.q[base + c % 16];
                                    i32::from(if c < 16 { b & 0x0F } else { b >> 4 }) - 8
                                }
                            };
                            gs = self.x[i * self.xs + kk].mul_add(code as f32, gs);
                        }
                        acc = gs.mul_add(self.s[p * self.ss + g * PANEL + c], acc);
                    }
                    unsafe { self.y.write(at, acc) };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{GROUP, PANEL, QTile};
    use std::arch::x86_64::*;

    /// Columns 0-15 and 16-31 of one panel at one position, as f32.
    #[inline(always)]
    unsafe fn load<const Q4: bool>(p: *const u8) -> [__m512; 2] {
        unsafe {
            if Q4 {
                let b = _mm512_cvtepu8_epi32(_mm_loadu_si128(p.cast()));
                let eight = _mm512_set1_epi32(8);
                let lo = _mm512_sub_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x0F)), eight);
                let hi = _mm512_sub_epi32(_mm512_srli_epi32(b, 4), eight);
                [_mm512_cvtepi32_ps(lo), _mm512_cvtepi32_ps(hi)]
            } else {
                [
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p.cast()))),
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p.add(16).cast()))),
                ]
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn tile<const MR: usize, const NP: usize, const Q4: bool>(t: &QTile) {
        unsafe {
            let x = t.x.as_ptr();
            let q = t.q.as_ptr();
            let bpk = if Q4 { PANEL / 2 } else { PANEL };
            if t.init {
                for i in 0..MR {
                    for p in 0..NP {
                        let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                        _mm512_storeu_ps(y, _mm512_setzero_ps());
                        _mm512_storeu_ps(y.add(16), _mm512_setzero_ps());
                    }
                }
            }
            for g in 0..t.kc / GROUP {
                let mut gs = [[[_mm512_setzero_ps(); 2]; NP]; MR];
                for kk in g * GROUP..(g + 1) * GROUP {
                    let mut xb = [_mm512_setzero_ps(); MR];
                    for (i, b) in xb.iter_mut().enumerate() {
                        *b = _mm512_set1_ps(*x.add(i * t.xs + kk));
                    }
                    #[allow(clippy::needless_range_loop)]
                    for p in 0..NP {
                        let w = load::<Q4>(q.add(p * t.qs + kk * bpk));
                        for i in 0..MR {
                            gs[i][p][0] = _mm512_fmadd_ps(xb[i], w[0], gs[i][p][0]);
                            gs[i][p][1] = _mm512_fmadd_ps(xb[i], w[1], gs[i][p][1]);
                        }
                    }
                }
                #[allow(clippy::needless_range_loop)]
                for p in 0..NP {
                    let sp = t.s.as_ptr().add(p * t.ss + g * PANEL);
                    let (s0, s1) = (_mm512_loadu_ps(sp), _mm512_loadu_ps(sp.add(16)));
                    for i in 0..MR {
                        let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                        _mm512_storeu_ps(y, _mm512_fmadd_ps(gs[i][p][0], s0, _mm512_loadu_ps(y)));
                        _mm512_storeu_ps(y.add(16), _mm512_fmadd_ps(gs[i][p][1], s1, _mm512_loadu_ps(y.add(16))));
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GROUP, PANEL, QTile};
    use std::arch::x86_64::*;

    /// Columns 0-7, 8-15, 16-23 and 24-31 of one panel at one position.
    #[inline(always)]
    unsafe fn load<const Q4: bool>(p: *const u8) -> [__m256; 4] {
        unsafe {
            if Q4 {
                let b0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(p.cast()));
                let b1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(p.add(8).cast()));
                let (m, eight) = (_mm256_set1_epi32(0x0F), _mm256_set1_epi32(8));
                let f = |v: __m256i| _mm256_cvtepi32_ps(_mm256_sub_epi32(v, eight));
                [
                    f(_mm256_and_si256(b0, m)),
                    f(_mm256_and_si256(b1, m)),
                    f(_mm256_srli_epi32(b0, 4)),
                    f(_mm256_srli_epi32(b1, 4)),
                ]
            } else {
                let f = |o: usize| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(p.add(o).cast())));
                [f(0), f(8), f(16), f(24)]
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile<const MR: usize, const NP: usize, const Q4: bool>(t: &QTile) {
        unsafe {
            let x = t.x.as_ptr();
            let q = t.q.as_ptr();
            let bpk = if Q4 { PANEL / 2 } else { PANEL };
            if t.init {
                for i in 0..MR {
                    for p in 0..NP {
                        let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                        for v in 0..4 {
                            _mm256_storeu_ps(y.add(8 * v), _mm256_setzero_ps());
                        }
                    }
                }
            }
            for g in 0..t.kc / GROUP {
                let mut gs = [[[_mm256_setzero_ps(); 4]; NP]; MR];
                for kk in g * GROUP..(g + 1) * GROUP {
                    let mut xb = [_mm256_setzero_ps(); MR];
                    for (i, b) in xb.iter_mut().enumerate() {
                        *b = _mm256_broadcast_ss(&*x.add(i * t.xs + kk));
                    }
                    #[allow(clippy::needless_range_loop)]
                    for p in 0..NP {
                        let w = load::<Q4>(q.add(p * t.qs + kk * bpk));
                        for i in 0..MR {
                            for v in 0..4 {
                                gs[i][p][v] = _mm256_fmadd_ps(xb[i], w[v], gs[i][p][v]);
                            }
                        }
                    }
                }
                #[allow(clippy::needless_range_loop)]
                for p in 0..NP {
                    let sp = t.s.as_ptr().add(p * t.ss + g * PANEL);
                    for i in 0..MR {
                        let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                        for v in 0..4 {
                            let s = _mm256_loadu_ps(sp.add(8 * v));
                            _mm256_storeu_ps(
                                y.add(8 * v),
                                _mm256_fmadd_ps(gs[i][p][v], s, _mm256_loadu_ps(y.add(8 * v))),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{Matrix, force_portable, matmul};
    use crate::pool::Pool;
    use crate::rng::Rng;
    use crate::safetensors::f32_to_bf16;

    fn setup(n: usize, k: usize, m: usize, seed: u64) -> (Matrix, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(rng.normal() as f32)).collect();
        let x: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
        (Matrix::new(n, k, w), x)
    }

    #[test]
    fn quantized_matmul_is_exact_on_its_own_weights_and_batch_invariant() {
        let pool = Pool::new(4);
        let (n, k, m) = (96, 544, 77);
        let (w, x) = setup(n, k, m, 11);
        for kind in [Quant::Int8, Quant::Int4] {
            let qw = w.quantized(kind, None);
            assert_eq!(qw.quant(), kind);
            let deq = qw.to_f32();
            let mut y = vec![0f32; m * n];
            matmul(&pool, &x, m, &qw, None, &mut y);
            // Against the dequantized weights in f64.
            for i in 0..m {
                for j in 0..n {
                    let exact: f64 = (0..k)
                        .map(|t| f64::from(x[i * k + t]) * f64::from(deq[j * k + t]))
                        .sum();
                    assert!((f64::from(y[i * n + j]) - exact).abs() < 2e-3, "{kind:?} {i},{j}");
                }
            }
            // Every batch size, one thread, and the portable path: same bits.
            for b in [1, 2, 3, 5, 8, 13, 61] {
                let mut yb = vec![0f32; b * n];
                matmul(&pool, &x[..b * k], b, &qw, None, &mut yb);
                assert_eq!(yb, y[..b * n], "{kind:?} batch {b}");
            }
            let mut y1 = vec![0f32; m * n];
            matmul(&Pool::new(1), &x, m, &qw, None, &mut y1);
            assert_eq!(y1, y, "{kind:?} thread count");
            force_portable(true);
            let mut yp = vec![0f32; m * n];
            matmul(&pool, &x, m, &qw, None, &mut yp);
            force_portable(false);
            assert_eq!(yp, y, "{kind:?} portable vs SIMD");
        }
    }

    #[test]
    fn quantization_error_is_small_and_clipping_helps() {
        let (n, k) = (32, 256);
        let (w, _) = setup(n, k, 1, 5);
        let orig = w.to_f32();
        for (kind, bound) in [(Quant::Int8, 0.006), (Quant::Int4, 0.12)] {
            let deq = w.quantized(kind, None).to_f32();
            let num: f64 = orig.iter().zip(&deq).map(|(a, b)| f64::from(a - b).powi(2)).sum();
            let den: f64 = orig.iter().map(|a| f64::from(*a).powi(2)).sum();
            let rel = (num / den).sqrt();
            assert!(rel < bound, "{kind:?}: relative error {rel}");
        }
        // Zero weights stay zero; int4 codes stay in range.
        let z = Matrix::new(32, 32, vec![0; 32 * 32]).quantized(Quant::Int4, None);
        assert!(z.to_f32().iter().all(|&v| v == 0.0));
    }
}
