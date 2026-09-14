//! ggml quant type block sizes and byte-per-element math.
//!
//! The fit checker (P1.3) needs the exact on-disk size of every tensor to
//! sum the model's weight bytes. ggml stores quantized tensors in fixed-size
//! *blocks*: e.g. Q4_0 packs 32 weights into 18 bytes. This module encodes
//! that table and derives `bytes_per_element` and `n_elements` for a tensor.

use crate::gguf::GgmlType;

/// Per-type encoding: block size (elements per block) and bytes per block.
/// `None` block size = 1 (per-element types, `bytes_per_element` is exact).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeInfo {
    pub block_size: u32,
    pub bytes_per_block: u32,
}

impl TypeInfo {
    /// Exact bytes per element (fractional for block-quantized types).
    pub fn bytes_per_element(self) -> f64 {
        self.bytes_per_block as f64 / self.block_size as f64
    }
}

/// Map a ggml type tag to its block encoding. Returns `None` for unknown
/// (future) types, which the estimator must then treat conservatively.
pub fn type_info(t: GgmlType) -> Option<TypeInfo> {
    use GgmlType::*;
    // (block_size, bytes_per_block)
    let (bs, bpb) = match t {
        F32 => (1, 4),
        F64 => (1, 8),
        F16 => (1, 2),
        BF16 => (1, 2),
        I8 => (1, 1),
        I16 => (1, 2),
        I32 => (1, 4),
        I64 => (1, 8),
        Q4_0 => (32, 18),
        Q4_1 => (32, 20),
        Q5_0 => (32, 22),
        Q5_1 => (32, 24),
        Q8_0 => (32, 34),
        Q8_1 => (32, 40),
        Q2_K => (256, 84),
        Q3_K => (256, 110),
        Q4_K => (256, 144),
        Q5_K => (256, 176),
        Q6_K => (256, 210),
        Q8_K => (256, 292),
        IQ2_XXS => (256, 66),
        IQ2_XS => (256, 74),
        IQ3_XXS => (256, 98),
        IQ3_S => (256, 110),
        IQ2_S => (256, 66),
        IQ4_XS => (256, 136),
        IQ1_S => (256, 74),
        IQ4_NL => (32, 18),
        IQ3_XS => (256, 110),
        IQ1_M => (256, 66),
        MXFP4 => (32, 17),
        TQ1_0 => (32, 22),
        TQ2_0 => (32, 24),
        Unknown(_) => return None,
    };
    Some(TypeInfo {
        block_size: bs,
        bytes_per_block: bpb,
    })
}

/// Number of elements in a tensor given its dimensions (product of dims).
pub fn n_elements(dims: &[u64]) -> u64 {
    dims.iter().product()
}

/// On-disk byte size of a tensor, computed exactly from its dims and type.
/// Rounds up to whole blocks (ggml stores whole blocks).
pub fn tensor_bytes(dims: &[u64], t: GgmlType) -> Option<u64> {
    let info = type_info(t)?;
    let n = n_elements(dims);
    let blocks = n.div_ceil(info.block_size as u64);
    Some(blocks * info.bytes_per_block as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_spec() {
        // The handful the plan explicitly pins (P1.3).
        assert_eq!(type_info(GgmlType::Q4_0).unwrap().bytes_per_block, 18);
        assert_eq!(type_info(GgmlType::Q8_0).unwrap().bytes_per_block, 34);
        assert_eq!(type_info(GgmlType::Q4_K).unwrap().bytes_per_block, 144);
        assert_eq!(type_info(GgmlType::Q4_K).unwrap().block_size, 256);
        assert_eq!(type_info(GgmlType::Q6_K).unwrap().bytes_per_block, 210);
        assert_eq!(type_info(GgmlType::MXFP4).unwrap().bytes_per_block, 17);
        assert_eq!(type_info(GgmlType::MXFP4).unwrap().block_size, 32);
    }

    #[test]
    fn f16_f32_exact() {
        assert_eq!(type_info(GgmlType::F16).unwrap().bytes_per_element(), 2.0);
        assert_eq!(type_info(GgmlType::F32).unwrap().bytes_per_element(), 4.0);
    }

    #[test]
    fn tensor_bytes_rounds_to_blocks() {
        // 33 elements of Q4_0 -> 2 blocks (18*2=36), not 33*18/32.
        let b = tensor_bytes(&[33], GgmlType::Q4_0).unwrap();
        assert_eq!(b, 36);
        // exactly 32 elements -> 1 block.
        assert_eq!(tensor_bytes(&[32], GgmlType::Q4_0).unwrap(), 18);
    }

    #[test]
    fn unknown_type_is_none() {
        assert!(type_info(GgmlType::Unknown(255)).is_none());
    }
}
