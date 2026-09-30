//! Reads Hugging Face safetensors checkpoints (single file or sharded with
//! an index). Only the JSON header is read eagerly; each tensor is read from
//! its offset on demand, so peak memory is the model, not model plus file.

use crate::json::{self, Json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    file: usize,
    offset: u64,
    len: usize,
}

pub struct Checkpoint {
    files: Vec<PathBuf>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl Checkpoint {
    /// Opens `dir/model.safetensors`, or the shards listed in
    /// `dir/model.safetensors.index.json`.
    pub fn open(dir: &Path) -> Result<Checkpoint, String> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<PathBuf> = if index.exists() {
            let text = std::fs::read_to_string(&index).map_err(|e| format!("{index:?}: {e}"))?;
            let v = json::parse(&text)?;
            let map = v
                .get("weight_map")
                .and_then(Json::as_obj)
                .ok_or("index without weight_map")?;
            let mut names: Vec<&str> = map.values().filter_map(Json::as_str).collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|n| dir.join(n)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };
        let mut tensors = BTreeMap::new();
        for (fi, path) in files.iter().enumerate() {
            let mut f = File::open(path).map_err(|e| format!("{path:?}: {e}"))?;
            let mut n = [0u8; 8];
            f.read_exact(&mut n).map_err(|e| format!("{path:?}: {e}"))?;
            let n = u64::from_le_bytes(n) as usize;
            if n > 100 << 20 {
                return Err(format!("{path:?}: implausible header size {n}"));
            }
            let mut header = vec![0u8; n];
            f.read_exact(&mut header).map_err(|e| format!("{path:?}: {e}"))?;
            let header = String::from_utf8(header).map_err(|_| "header is not UTF-8")?;
            let v = json::parse(&header)?;
            for (name, t) in v.as_obj().ok_or("header is not an object")? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match t.get("dtype").and_then(Json::as_str) {
                    Some("F32") => DType::F32,
                    Some("F16") => DType::F16,
                    Some("BF16") => DType::BF16,
                    // Integer buffers (e.g. BERT's position_ids) are not
                    // weights; skip them.
                    _ => continue,
                };
                let shape: Vec<usize> = t
                    .get("shape")
                    .map(|s| s.as_arr().iter().filter_map(Json::as_usize).collect())
                    .unwrap_or_default();
                let off = t.get("data_offsets").map(Json::as_arr).unwrap_or(&[]);
                let (Some(a), Some(b)) = (
                    off.first().and_then(Json::as_usize),
                    off.get(1).and_then(Json::as_usize),
                ) else {
                    return Err(format!("{name}: bad data_offsets"));
                };
                let elems: usize = shape.iter().product();
                if b < a || b - a != elems * dtype.size() {
                    return Err(format!("{name}: {} bytes for shape {shape:?}", b - a));
                }
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype,
                        shape,
                        file: fi,
                        offset: (8 + n + a) as u64,
                        len: b - a,
                    },
                );
            }
        }
        Ok(Checkpoint { files, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("checkpoint has no tensor {name}"))
    }

    fn raw(&self, name: &str) -> Result<(&TensorInfo, Vec<u8>), String> {
        let t = self.info(name)?;
        let path = &self.files[t.file];
        let mut f = File::open(path).map_err(|e| format!("{path:?}: {e}"))?;
        f.seek(SeekFrom::Start(t.offset))
            .map_err(|e| format!("{path:?}: {e}"))?;
        let mut buf = vec![0u8; t.len];
        f.read_exact(&mut buf).map_err(|e| format!("{name}: {e}"))?;
        Ok((t, buf))
    }

    /// The tensor as bf16 bit patterns. bf16 checkpoints load exactly;
    /// f32 and f16 are rounded to nearest even.
    pub fn bf16(&self, name: &str) -> Result<Vec<u16>, String> {
        let (t, b) = self.raw(name)?;
        Ok(match t.dtype {
            DType::BF16 => b.as_chunks::<2>().0.iter().map(|&c| u16::from_le_bytes(c)).collect(),
            _ => to_f32(t.dtype, &b).into_iter().map(f32_to_bf16).collect(),
        })
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let (t, b) = self.raw(name)?;
        Ok(to_f32(t.dtype, &b))
    }
}

fn to_f32(dtype: DType, b: &[u8]) -> Vec<f32> {
    match dtype {
        DType::F32 => b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        DType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16_to_f32(u16::from_le_bytes(c)))
            .collect(),
        DType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| f16_to_f32(u16::from_le_bytes(c)))
            .collect(),
    }
}

pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

/// Round to nearest, ties to even; NaN stays NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let round = 0x7FFF + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let man = u32::from(h & 0x3FF);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalise the mantissa.
            let shift = m.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((m << shift) & 0x3FF) << 13)
        }
        (0x1F, m) => sign | 0x7F80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Writes a safetensors file (used by tests to build small random models).
pub fn write(path: &Path, tensors: &[(&str, DType, Vec<usize>, Vec<u8>)]) -> std::io::Result<()> {
    let mut header = String::from("{");
    let mut off = 0;
    for (i, (name, dtype, shape, data)) in tensors.iter().enumerate() {
        let d = match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        };
        let shape: Vec<String> = shape.iter().map(ToString::to_string).collect();
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "{}:{{\"dtype\":\"{d}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
            json::quote(name),
            shape.join(","),
            off + data.len()
        ));
        off += data.len();
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    for t in tensors {
        out.extend_from_slice(&t.3);
    }
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(bf16_to_f32(f32_to_bf16(3.140625)), 3.140625);
        // Exactly halfway between two bf16 values rounds to the even one.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_decodes_normals_subnormals_and_specials() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("ferrolm-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a: Vec<u8> = [1.0f32, -2.5, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let b: Vec<u8> = [0x3F80u16, 0x4000].iter().flat_map(|x| x.to_le_bytes()).collect();
        write(
            &dir.join("model.safetensors"),
            &[("a", DType::F32, vec![3], a), ("b.w", DType::BF16, vec![1, 2], b)],
        )
        .unwrap();
        let c = Checkpoint::open(&dir).unwrap();
        assert_eq!(c.f32("a").unwrap(), vec![1.0, -2.5, 3.0]);
        assert_eq!(c.f32("b.w").unwrap(), vec![1.0, 2.0]);
        assert_eq!(c.bf16("a").unwrap()[1], f32_to_bf16(-2.5));
        assert_eq!(c.info("b.w").unwrap().shape, vec![1, 2]);
        std::fs::remove_dir_all(dir).ok();
    }
}
