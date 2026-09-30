//! The `.fvecs`/`.ivecs` files of the classic ANN benchmarks (SIFT1M,
//! GIST1M): each vector is a little-endian i32 dimension followed by that
//! many f32 (or i32) values.

use std::path::Path;

fn read_vecs(path: &Path, limit: Option<usize>) -> Result<(Vec<[u8; 4]>, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    if bytes.len() < 4 {
        return Err(format!("{path:?}: empty"));
    }
    let dim = i32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let rec = 4 + 4 * dim;
    if dim == 0 || bytes.len() % rec != 0 {
        return Err(format!("{path:?}: not a vecs file of dimension {dim}"));
    }
    let n = (bytes.len() / rec).min(limit.unwrap_or(usize::MAX));
    let mut out = Vec::with_capacity(n * dim);
    for r in 0..n {
        let base = r * rec;
        if i32::from_le_bytes(bytes[base..base + 4].try_into().unwrap()) as usize != dim {
            return Err(format!("{path:?}: record {r} has a different dimension"));
        }
        out.extend(bytes[base + 4..base + rec].as_chunks::<4>().0.iter().copied());
    }
    Ok((out, dim))
}

/// Up to `limit` vectors, flattened, and their dimension.
pub fn read_fvecs(path: &Path, limit: Option<usize>) -> Result<(Vec<f32>, usize), String> {
    let (raw, dim) = read_vecs(path, limit)?;
    Ok((raw.into_iter().map(f32::from_le_bytes).collect(), dim))
}

/// Up to `limit` rows of integers (e.g. ground-truth neighbour ids).
pub fn read_ivecs(path: &Path, limit: Option<usize>) -> Result<Vec<Vec<u32>>, String> {
    let (raw, dim) = read_vecs(path, limit)?;
    Ok(raw
        .chunks_exact(dim)
        .map(|r| r.iter().map(|b| i32::from_le_bytes(*b) as u32).collect())
        .collect())
}

/// A little-endian binary writer and reader for saving indexes.
pub struct Writer(pub Vec<u8>);

impl Writer {
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    pub fn u32s(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    pub fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
    }
}

pub struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, i: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.i..self.i + n).ok_or("index file is truncated")?;
        self.i += n;
        Ok(s)
    }
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.u64()? as usize;
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
    pub fn u32s(&mut self) -> Result<Vec<u32>, String> {
        let n = self.u64()? as usize;
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }
    pub fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let n = self.u64()? as usize;
        Ok(self.take(n)?.to_vec())
    }
}
