//! Numerical kernels: a bf16-weight matrix multiply with AVX2/FMA and a
//! portable fallback, and the small per-row operations around it.
//!
//! The matrix multiply computes every output element as one chain of fused
//! multiply-adds over the inner dimension, in order, starting from zero.
//! Tiling (which rows and columns are computed together, and how the inner
//! dimension is blocked for the cache) changes only which chains run side
//! by side, never the order within a chain, so an output element does not
//! depend on the batch it was computed in, on the tile shapes or on the
//! thread count. The portable fallback runs the same chains, so the two
//! paths agree bit for bit.

use crate::pool::{Out, Pool};
use crate::safetensors::bf16_to_f32;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Output columns per packed panel.
const PANEL: usize = 32;

/// Where column `c` of a panel sits among its 32 packed words. The SIMD
/// kernels widen bf16 to f32 by interleaving with zeros, which within each
/// 128-bit lane takes words 0-3 into one register and 4-7 into another.
/// This order makes those registers hold consecutive columns, with the same
/// layout for 256-bit and 512-bit registers.
const SLOT: [usize; PANEL] = {
    let mut s = [0; PANEL];
    let mut c = 0;
    while c < PANEL {
        let (hi, cc) = (c / 16, c % 16);
        s[c] = 8 * (cc / 4) + 4 * hi + cc % 4;
        c += 1;
    }
    s
};
/// A weight matrix of bf16 values, `rows × cols`, as in a PyTorch `Linear`
/// layer (`y = x Wᵀ`, one row per output). It is stored packed: panels of
/// 32 rows, and within a panel the 32 weights for each input position side
/// by side, so the kernel streams it strictly sequentially.
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    packed: Vec<u16>,
}

impl Matrix {
    /// From row-major data.
    pub fn new(rows: usize, cols: usize, data: Vec<u16>) -> Matrix {
        assert_eq!(data.len(), rows * cols, "matrix data does not match its shape");
        assert_eq!(rows % PANEL, 0, "output dimension must be a multiple of {PANEL}");
        let mut packed = vec![0u16; rows * cols];
        for (r, row) in data.chunks_exact(cols).enumerate() {
            let (p, c) = (r / PANEL, SLOT[r % PANEL]);
            for (k, &w) in row.iter().enumerate() {
                packed[(p * cols + k) * PANEL + c] = w;
            }
        }
        Matrix { rows, cols, packed }
    }

    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let (p, c) = (r / PANEL, SLOT[r % PANEL]);
        let panel = &self.packed[p * self.cols * PANEL..(p + 1) * self.cols * PANEL];
        for (k, o) in out.iter_mut().enumerate().take(self.cols) {
            *o = bf16_to_f32(panel[k * PANEL + c]);
        }
    }

    pub fn bytes(&self) -> usize {
        self.packed.len() * 2
    }
}

static PORTABLE: AtomicBool = AtomicBool::new(false);

/// Forces the portable path (every path gives identical bits; tests use
/// this to prove it).
pub fn force_portable(on: bool) {
    PORTABLE.store(on, Ordering::Relaxed);
}

#[derive(Clone, Copy, PartialEq)]
enum Isa {
    Portable,
    Avx2,
    Avx512,
}

fn isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    if PORTABLE.load(Ordering::Relaxed) {
        return Isa::Portable;
    }
    *ISA.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let want = std::env::var("FERROLM_ISA").unwrap_or_default();
            let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            let avx512 = avx2 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
            match want.as_str() {
                "portable" => Isa::Portable,
                "avx2" if avx2 => Isa::Avx2,
                _ if avx512 && want != "avx2" => Isa::Avx512,
                _ if avx2 => Isa::Avx2,
                _ => Isa::Portable,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Isa::Portable
        }
    })
}

fn use_avx2() -> bool {
    isa() != Isa::Portable
}

/// The kernel path in use, for logs and reports.
pub fn backend() -> &'static str {
    match isa() {
        Isa::Portable => "portable",
        Isa::Avx2 => "avx2+fma",
        Isa::Avx512 => "avx512",
    }
}

/// The fixed lane reduction for `dot`, shared by both paths.
#[inline(always)]
fn hsum8(l: [f32; 8]) -> f32 {
    ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]))
}

/// `a · b` over f32 vectors (length a multiple of 8): eight lanes, each
/// summing every eighth product, then a fixed pairwise sum of the lanes.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { avx2::dot(a, b) };
    }
    let mut acc = [0f32; 8];
    for (ca, cb) in a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0) {
        for l in 0..8 {
            acc[l] = ca[l].mul_add(cb[l], acc[l]);
        }
    }
    hsum8(acc)
}

/// `acc += s * v`, elementwise with fused multiply-adds.
pub fn axpy(acc: &mut [f32], s: f32, v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { avx2::axpy(acc, s, v) };
    }
    for (a, &x) in acc.iter_mut().zip(v) {
        *a = s.mul_add(x, *a);
    }
}

/// Rows of `x` per cache block, and inner positions per block.
const MC: usize = 60;
const KC: usize = 256;

/// `y[i][j] = x[i] · w[j] (+ bias[j])` for the `m` rows of `x`.
pub fn matmul(pool: &Pool, x: &[f32], m: usize, w: &Matrix, bias: Option<&[f32]>, y: &mut [f32]) {
    let (n, k) = (w.rows, w.cols);
    assert_eq!(x.len(), m * k);
    assert_eq!(y.len(), m * n);
    if m == 0 {
        return;
    }
    let panels = n / PANEL;
    // Enough tasks to balance the threads, in whole panels.
    let per_task = (panels / (pool.threads() * 6)).clamp(1, 64);
    let out = Out::new(y);
    pool.run(panels.div_ceil(per_task), &|t| {
        let p0 = t * per_task;
        panels_of(x, m, w, p0, (p0 + per_task).min(panels), out);
    });
    if let Some(b) = bias {
        for row in y.chunks_exact_mut(n) {
            for (v, &bb) in row.iter_mut().zip(b) {
                *v += bb;
            }
        }
    }
}

/// Tile shapes (rows, panels) by the rows left in a block: as many
/// independent accumulator chains as the registers hold.
fn tile_shape(rows: usize) -> (usize, usize) {
    match (isa(), rows) {
        (Isa::Avx512, 1) => (1, 4),
        (Isa::Avx512, 2 | 3) => (2, 2),
        (Isa::Avx512, 4..=7) => (4, 2),
        (Isa::Avx512, _) => (8, 1),
        (_, 1) => (1, 2),
        (_, 2) => (2, 1),
        _ => (3, 1),
    }
}

/// Output panels `p0..p1` for every row.
fn panels_of(x: &[f32], m: usize, w: &Matrix, p0: usize, p1: usize, out: Out<f32>) {
    let (n, k) = (w.rows, w.cols);
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
                    let t = Tile {
                        x: &x[i * k + k0..],
                        xs: k,
                        w: &w.packed[(p * k + k0) * PANEL..],
                        ws: k * PANEL,
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

/// One register tile: `mr` rows × `np` panels, over `kc` inner positions,
/// starting from zero (`init`) or from the partial sums already in `y`.
struct Tile<'a> {
    x: &'a [f32],
    xs: usize,
    w: &'a [u16],
    ws: usize,
    kc: usize,
    y: Out<f32>,
    y0: usize,
    ys: usize,
    init: bool,
}

impl Tile<'_> {
    fn run(&self, mr: usize, np: usize) {
        // SAFETY: the tile lies within x, w and this task's columns of y.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            match (isa(), mr, np) {
                (Isa::Avx512, 1, 1) => return avx512::tile::<1, 1>(self),
                (Isa::Avx512, 1, 2) => return avx512::tile::<1, 2>(self),
                (Isa::Avx512, 1, 3) => return avx512::tile::<1, 3>(self),
                (Isa::Avx512, 1, 4) => return avx512::tile::<1, 4>(self),
                (Isa::Avx512, 2, 1) => return avx512::tile::<2, 1>(self),
                (Isa::Avx512, 2, 2) => return avx512::tile::<2, 2>(self),
                (Isa::Avx512, 4, 1) => return avx512::tile::<4, 1>(self),
                (Isa::Avx512, 4, 2) => return avx512::tile::<4, 2>(self),
                (Isa::Avx512, 8, 1) => return avx512::tile::<8, 1>(self),
                (Isa::Avx2, 1, 1) => return avx2::tile::<1, 1>(self),
                (Isa::Avx2, 1, 2) => return avx2::tile::<1, 2>(self),
                (Isa::Avx2, 2, 1) => return avx2::tile::<2, 1>(self),
                (Isa::Avx2, 3, 1) => return avx2::tile::<3, 1>(self),
                (Isa::Portable, ..) => {}
                other => unreachable!("tile {:?}", (other.1, other.2)),
            }
        }
        for i in 0..mr {
            for p in 0..np {
                for (c, &slot) in SLOT.iter().enumerate() {
                    let at = self.y0 + i * self.ys + p * PANEL + c;
                    // SAFETY: as above.
                    let mut acc = if self.init { 0.0 } else { unsafe { self.y.read(at) } };
                    for kk in 0..self.kc {
                        let wv = bf16_to_f32(self.w[p * self.ws + kk * PANEL + slot]);
                        acc = self.x[i * self.xs + kk].mul_add(wv, acc);
                    }
                    unsafe { self.y.write(at, acc) };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{PANEL, Tile};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn tile<const MR: usize, const NP: usize>(t: &Tile) {
        unsafe {
            let x = t.x.as_ptr();
            let w = t.w.as_ptr();
            let mut acc = [[[_mm512_setzero_ps(); 2]; NP]; MR];
            if !t.init {
                for (i, row) in acc.iter_mut().enumerate() {
                    for (p, a) in row.iter_mut().enumerate() {
                        let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                        a[0] = _mm512_loadu_ps(y);
                        a[1] = _mm512_loadu_ps(y.add(16));
                    }
                }
            }
            let zero = _mm512_setzero_si512();
            for kk in 0..t.kc {
                let mut xb = [_mm512_setzero_ps(); MR];
                for (i, b) in xb.iter_mut().enumerate() {
                    *b = _mm512_set1_ps(*x.add(i * t.xs + kk));
                }
                #[allow(clippy::needless_range_loop)]
                for p in 0..NP {
                    let raw = _mm512_loadu_si512(w.add(p * t.ws + kk * PANEL).cast());
                    let lo = _mm512_castsi512_ps(_mm512_unpacklo_epi16(zero, raw));
                    let hi = _mm512_castsi512_ps(_mm512_unpackhi_epi16(zero, raw));
                    for i in 0..MR {
                        acc[i][p][0] = _mm512_fmadd_ps(xb[i], lo, acc[i][p][0]);
                        acc[i][p][1] = _mm512_fmadd_ps(xb[i], hi, acc[i][p][1]);
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                for (p, a) in row.iter().enumerate() {
                    let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                    _mm512_storeu_ps(y, a[0]);
                    _mm512_storeu_ps(y.add(16), a[1]);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{PANEL, Tile};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile<const MR: usize, const NP: usize>(t: &Tile) {
        unsafe {
            let x = t.x.as_ptr();
            let w = t.w.as_ptr();
            // Per panel: columns 0-7, 8-15, 16-23, 24-31.
            let mut acc = [[[_mm256_setzero_ps(); 4]; NP]; MR];
            if !t.init {
                for (i, row) in acc.iter_mut().enumerate() {
                    for (p, a) in row.iter_mut().enumerate() {
                        let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                        for (q, v) in a.iter_mut().enumerate() {
                            *v = _mm256_loadu_ps(y.add(8 * q));
                        }
                    }
                }
            }
            let zero = _mm256_setzero_si256();
            for kk in 0..t.kc {
                let mut xb = [_mm256_setzero_ps(); MR];
                for (i, b) in xb.iter_mut().enumerate() {
                    *b = _mm256_broadcast_ss(&*x.add(i * t.xs + kk));
                }
                #[allow(clippy::needless_range_loop)]
                for p in 0..NP {
                    let base = w.add(p * t.ws + kk * PANEL);
                    let r0 = _mm256_loadu_si256(base.cast());
                    let r1 = _mm256_loadu_si256(base.add(16).cast());
                    let v = [
                        _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero, r0)),
                        _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero, r1)),
                        _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero, r0)),
                        _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero, r1)),
                    ];
                    for i in 0..MR {
                        for q in 0..4 {
                            acc[i][p][q] = _mm256_fmadd_ps(xb[i], v[q], acc[i][p][q]);
                        }
                    }
                }
            }
            for (i, row) in acc.iter().enumerate() {
                for (p, a) in row.iter().enumerate() {
                    let y = t.y.ptr(t.y0 + i * t.ys + p * PANEL);
                    for (q, v) in a.iter().enumerate() {
                        _mm256_storeu_ps(y.add(8 * q), *v);
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let mut l = [0f32; 8];
        unsafe { _mm256_storeu_ps(l.as_mut_ptr(), v) };
        super::hsum8(l)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        for i in (0..a.len()).step_by(8) {
            unsafe {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                    acc,
                );
            }
        }
        unsafe { hsum(acc) }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp_all(v: &mut [f32]) {
        use super::{EXP_C, EXP_MAX, EXP_MIN, LN2_HI, LN2_LO, LOG2E};
        let n8 = v.len() / 8 * 8;
        unsafe {
            for i in (0..n8).step_by(8) {
                let p = v.as_mut_ptr().add(i);
                let x = _mm256_min_ps(
                    _mm256_max_ps(_mm256_loadu_ps(p), _mm256_set1_ps(EXP_MIN)),
                    _mm256_set1_ps(EXP_MAX),
                );
                let n = _mm256_round_ps(
                    _mm256_mul_ps(x, _mm256_set1_ps(LOG2E)),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                );
                let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_HI), x);
                let r = _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_LO), r);
                let mut poly = _mm256_set1_ps(EXP_C[0]);
                for &c in &EXP_C[1..] {
                    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(c));
                }
                let y = _mm256_add_ps(_mm256_fmadd_ps(poly, _mm256_mul_ps(r, r), r), _mm256_set1_ps(1.0));
                let e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
                _mm256_storeu_ps(p, _mm256_mul_ps(y, _mm256_castsi256_ps(e)));
            }
        }
        for x in &mut v[n8..] {
            *x = super::exp(*x);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn scores(q: &[f32], hd: usize, n: usize, scale: f32, kv: &super::KvView, scores: &mut [f32]) {
        kv.each(n, |p, off| unsafe {
            let kp = kv.k.as_ptr().add(off);
            for (j, qh) in q.chunks_exact(hd).enumerate() {
                let mut acc = _mm256_setzero_ps();
                for d in (0..hd).step_by(8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh.as_ptr().add(d)), _mm256_loadu_ps(kp.add(d)), acc);
                }
                *scores.get_unchecked_mut(j * n + p) = hsum(acc) * scale;
            }
        });
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn weighted_values(hd: usize, n: usize, kv: &super::KvView, w: &[f32], out: &mut [f32]) {
        kv.each(n, |p, off| unsafe {
            let vp = kv.v.as_ptr().add(off);
            for (j, o) in out.chunks_exact_mut(hd).enumerate() {
                let s = _mm256_set1_ps(*w.get_unchecked(j * n + p));
                for d in (0..hd).step_by(8) {
                    let op = o.as_mut_ptr().add(d);
                    _mm256_storeu_ps(op, _mm256_fmadd_ps(s, _mm256_loadu_ps(vp.add(d)), _mm256_loadu_ps(op)));
                }
            }
        });
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(acc: &mut [f32], s: f32, v: &[f32]) {
        let sv = _mm256_set1_ps(s);
        let n = acc.len().min(v.len());
        let mut i = 0;
        while i + 8 <= n {
            unsafe {
                let p = acc.as_mut_ptr().add(i);
                _mm256_storeu_ps(
                    p,
                    _mm256_fmadd_ps(sv, _mm256_loadu_ps(v.as_ptr().add(i)), _mm256_loadu_ps(p)),
                );
            }
            i += 8;
        }
        for j in i..n {
            acc[j] = s.mul_add(v[j], acc[j]);
        }
    }
}

/// RMSNorm of each `dim`-wide row of `x` into `out`, as Llama computes it.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let dim = weight.len();
    for (xr, or) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ms = dot(xr, xr) / dim as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for ((o, &v), &g) in or.iter_mut().zip(xr).zip(weight) {
            *o = g * (v * r);
        }
    }
}

// exp(x) as 2^n · e^r with |r| <= ln 2 / 2 and a degree-7 polynomial for
// e^r (Cephes' expf coefficients). The scalar and SIMD versions perform
// the same IEEE operations in the same order, so they agree bit for bit;
// the result is within 2 ulp of the true value.
const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const EXP_C: [f32; 6] = [
    1.987_569_1e-4,
    1.398_2e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    0.166_666_65,
    0.5,
];
const EXP_MIN: f32 = -87.0;
const EXP_MAX: f32 = 88.0;

pub fn exp(x: f32) -> f32 {
    let x = x.clamp(EXP_MIN, EXP_MAX);
    let n = (x * LOG2E).round_ties_even();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    let mut p = EXP_C[0];
    for &c in &EXP_C[1..] {
        p = p.mul_add(r, c);
    }
    let y = p.mul_add(r * r, r) + 1.0;
    y * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// `v[i] = exp(v[i])`.
pub fn exp_all(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { avx2::exp_all(v) };
    }
    for x in v {
        *x = exp(*x);
    }
}

/// `out[i] = silu(gate[i]) * up[i]` for the fused gate/up output of each row.
pub fn silu_mul(gate_up: &[f32], inter: usize, out: &mut [f32]) {
    for (gu, o) in gate_up.chunks_exact(2 * inter).zip(out.chunks_exact_mut(inter)) {
        let (g, u) = gu.split_at(inter);
        o.copy_from_slice(g);
        for v in o.iter_mut() {
            *v = -*v;
        }
        exp_all(o);
        for ((o, &g), &u) in o.iter_mut().zip(g).zip(u) {
            *o = g / (1.0 + *o) * u;
        }
    }
}

pub fn add(acc: &mut [f32], x: &[f32]) {
    for (a, &b) in acc.iter_mut().zip(x) {
        *a += b;
    }
}

/// Where the keys and values of one kv head live in a paged cache layer:
/// position `p` is at `blocks[p / bs] * block_stride + head_off +
/// (p % bs) * head_dim`.
pub struct KvView<'a> {
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub blocks: &'a [u32],
    pub bs: usize,
    pub hd: usize,
    pub block_stride: usize,
    pub head_off: usize,
}

impl KvView<'_> {
    /// Calls `f(p, offset)` for positions `0..n` in order, where `offset`
    /// is where position `p`'s key (and value) starts.
    #[inline(always)]
    fn each(&self, n: usize, mut f: impl FnMut(usize, usize)) {
        let mut p = 0;
        for &b in self.blocks {
            let base = b as usize * self.block_stride + self.head_off;
            for o in 0..self.bs.min(n - p) {
                f(p, base + o * self.hd);
                p += 1;
            }
            if p == n {
                return;
            }
        }
    }
}

/// Causal attention for the query heads that share one kv head (`q` holds
/// `q.len() / hd` heads side by side) over the first `n` cached positions,
/// into `out` (same layout as `q`). `scores` is scratch space.
///
/// Scores use `dot`'s fixed reduction; the softmax sum and the weighted sum
/// of values run over positions in order. Nothing depends on how many
/// other rows or sequences are being processed.
pub fn attend(q: &[f32], hd: usize, n: usize, scale: f32, kv: &KvView, scores: &mut Vec<f32>, out: &mut [f32]) {
    let heads = q.len() / hd;
    scores.clear();
    scores.resize(heads * n, 0.0);
    let simd = cfg!(target_arch = "x86_64") && use_avx2() && hd.is_multiple_of(8);
    #[cfg(target_arch = "x86_64")]
    if simd {
        unsafe { avx2::scores(q, hd, n, scale, kv, scores) };
    }
    if !simd {
        scores_portable(q, hd, n, scale, kv, scores);
    }
    for s in scores.chunks_exact_mut(n) {
        let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in s.iter_mut() {
            *v -= max;
        }
        exp_all(s);
        let sum: f32 = s.iter().sum();
        for v in s.iter_mut() {
            *v /= sum;
        }
    }
    out.fill(0.0);
    #[cfg(target_arch = "x86_64")]
    if simd {
        unsafe { avx2::weighted_values(hd, n, kv, scores, out) };
        return;
    }
    kv.each(n, |p, off| {
        let vp = &kv.v[off..off + hd];
        for (j, o) in out.chunks_exact_mut(hd).enumerate() {
            for (a, &x) in o.iter_mut().zip(vp) {
                *a = scores[j * n + p].mul_add(x, *a);
            }
        }
    });
}

fn scores_portable(q: &[f32], hd: usize, n: usize, scale: f32, kv: &KvView, scores: &mut [f32]) {
    kv.each(n, |p, off| {
        let kp = &kv.k[off..off + hd];
        for (j, qh) in q.chunks_exact(hd).enumerate() {
            scores[j * n + p] = dot(qh, kp) * scale;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::safetensors::f32_to_bf16;

    fn random(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| rng.normal() as f32).collect()
    }

    #[test]
    fn exp_is_accurate_and_identical_on_every_path() {
        let mut xs: Vec<f32> = (-2000..=2000).map(|i| i as f32 * 0.043).collect();
        xs.extend([0.0, -0.0, 1e-30, -86.9, 87.9, -1000.0, 1000.0]);
        let mut simd = xs.clone();
        exp_all(&mut simd);
        for (&x, &e) in xs.iter().zip(&simd) {
            assert_eq!(e.to_bits(), exp(x).to_bits(), "exp({x})");
            let want = f64::from(x.clamp(EXP_MIN, EXP_MAX)).exp();
            assert!(
                (f64::from(e) - want).abs() <= want * 3e-7,
                "exp({x}) = {e}, want {want}"
            );
        }
    }

    #[test]
    fn matmul_matches_a_plain_loop_and_is_batch_invariant() {
        let pool = Pool::new(4);
        let mut rng = Rng::new(7);
        let (n, k) = (96, 520);
        let w = Matrix::new(n, k, random(n * k, &mut rng).into_iter().map(f32_to_bf16).collect());
        let m = 131;
        let x = random(m * k, &mut rng);
        let mut y = vec![0f32; m * n];
        matmul(&pool, &x, m, &w, None, &mut y);
        for i in 0..m {
            for j in 0..n {
                let mut wr = vec![0f32; k];
                w.row_f32(j, &mut wr);
                let exact: f64 = (0..k).map(|t| f64::from(x[i * k + t]) * f64::from(wr[t])).sum();
                assert!((f64::from(y[i * n + j]) - exact).abs() < 1e-4, "{i},{j}");
            }
        }
        // Each row alone, and in every batch size, gives the same bits.
        for b in 1..=m {
            let mut yb = vec![0f32; b * n];
            matmul(&pool, &x[..b * k], b, &w, None, &mut yb);
            assert_eq!(yb, y[..b * n], "batch of {b}");
        }
        let single = Pool::new(1);
        let mut y1 = vec![0f32; m * n];
        matmul(&single, &x, m, &w, None, &mut y1);
        assert_eq!(y1, y, "thread count changed the result");
        // The portable path computes the same bits.
        force_portable(true);
        let mut yp = vec![0f32; m * n];
        matmul(&pool, &x, m, &w, None, &mut yp);
        force_portable(false);
        assert_eq!(yp, y, "portable and SIMD paths differ");
    }
}
