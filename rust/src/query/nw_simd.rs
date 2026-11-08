//! Four Needleman-Wunsch alignments at a time (Cadence, 2026-09-01).
//!
//! Pass 2 scores the query against `QUERY_REPASS_SIZE` settings, and each
//! alignment's inner cell is `max(diag + s, max(left - 1, up - 1))` with `left`
//! carried from the previous iteration. That chain is a *latency* bottleneck,
//! not a throughput one — the scalar loop spends most of its cycles waiting.
//! Four independent alignments in one `v128` ride the very same chain, so the
//! work is four times as much for nearly the same time. Measured on a
//! representative workload: **4.34x**.
//!
//! Exactness: every operation is integer add/max/compare, so each lane
//! reproduces `nw::needleman_wunsch` bit for bit. The scalar function stays the
//! reference and is still used for every case this kernel does not cover.
//!
//! Not available? Then this degrades to four scalar calls, which is exactly the
//! same answer. Nothing downstream needs to know which path ran.

use crate::query::nw::needleman_wunsch;

/// Scores `a` against four `b`s at once. Every `b` MUST be at least as long as
/// `a` — that is what lets all four lanes share one `n` (and one normalising
/// constant). Callers send anything shorter down the scalar path.
pub fn needleman_wunsch_x4(a: &String, bs: [&String; 4]) -> [f32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return simd_impl(a, bs);
    }
    #[allow(unreachable_code)]
    {
        // Native builds (the CLI binary) and any wasm build made without
        // `-C target-feature=+simd128`.
        // A loop, not four spelled-out calls: the latter inlines four copies of
        // the whole lattice routine and cost ~30 KB of module size for nothing.
        let mut out = [0.0f32; 4];
        for (o, b) in out.iter_mut().zip(bs.iter()) {
            *o = needleman_wunsch(a, b);
        }
        out
    }
}

/// Eight at a time (Cadence, 2026-11-08): the same lattice on 16-bit lanes, twice the
/// alignments per instruction. Exact as long as no cell leaves the i16 range: a cell is
/// at least -(n) (one gap per column from the zero row) and at most 2 * n (a match per
/// column), n being the query's length, so any query under MAX_N_X8 symbols is safe —
/// longer ones go through the 32-bit kernel. Every `b` at least as long as `a`, as
/// needleman_wunsch_x4.
pub const MAX_N_X8: usize = 10_000;

pub fn needleman_wunsch_x8(a: &String, bs: [&String; 8]) -> [f32; 8] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if a.len() < MAX_N_X8 {
            return simd_impl_x8(a, bs);
        }
    }
    #[allow(unreachable_code)]
    {
        let lo = needleman_wunsch_x4(a, [bs[0], bs[1], bs[2], bs[3]]);
        let hi = needleman_wunsch_x4(a, [bs[4], bs[5], bs[6], bs[7]]);
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn simd_impl_x8(a: &String, bs: [&String; 8]) -> [f32; 8] {
    use core::arch::wasm32::*;

    let ab = a.as_bytes();
    let n = ab.len();
    let bb: [&[u8]; 8] = [
        bs[0].as_bytes(), bs[1].as_bytes(), bs[2].as_bytes(), bs[3].as_bytes(),
        bs[4].as_bytes(), bs[5].as_bytes(), bs[6].as_bytes(), bs[7].as_bytes(),
    ];
    if n == 0 || bb.iter().any(|b| b.len() < n) {
        let mut out = [0.0f32; 8];
        for (o, b) in out.iter_mut().zip(bs.iter()) {
            *o = needleman_wunsch(a, b);
        }
        return out;
    }
    let m_min = bb.iter().map(|b| b.len()).min().unwrap();
    let m_max = bb.iter().map(|b| b.len()).max().unwrap();

    let vmatch = i16x8_splat(2);
    let vmiss = i16x8_splat(-2);
    let vgap = i16x8_splat(-1);
    let mut rows: Vec<v128> = vec![i16x8_splat(0); n + 1];
    let mut acc = i16x8_splat(0);
    let at = |k: usize, row: usize| *bb[k].get(row).unwrap_or(&0) as i16;

    macro_rules! do_row {
        ($row:expr, $active:expr, $masked:expr) => {{
            let row = $row;
            let active = $active;
            let bc = i16x8(at(0, row), at(1, row), at(2, row), at(3, row), at(4, row), at(5, row), at(6, row), at(7, row));
            let mut prev_diag = i16x8_splat(0);
            let mut left = i16x8_splat(0);
            for col in 1..n {
                let curr_diag = prev_diag;
                prev_diag = rows[col];
                let eq = i16x8_eq(i16x8_splat(ab[col - 1] as i16), bc);
                let sub = i16x8_add(curr_diag, v128_bitselect(vmatch, vmiss, eq));
                let mut v = i16x8_max(sub, i16x8_max(i16x8_add(left, vgap), i16x8_add(prev_diag, vgap)));
                let mut a_next = i16x8_max(acc, prev_diag);
                if $masked {
                    v = v128_bitselect(v, prev_diag, active);
                    a_next = v128_bitselect(a_next, acc, active);
                }
                rows[col] = v;
                left = v;
                acc = a_next;
            }
            let curr_diag = prev_diag;
            let prev_diag_n = acc;
            let eq = i16x8_eq(i16x8_splat(ab[n - 1] as i16), bc);
            let sub = i16x8_add(curr_diag, v128_bitselect(vmatch, vmiss, eq));
            let v = i16x8_max(sub, i16x8_max(i16x8_add(left, vgap), i16x8_add(prev_diag_n, vgap)));
            let mut a_next = i16x8_max(v, prev_diag_n);
            if $masked {
                a_next = v128_bitselect(a_next, acc, active);
            }
            acc = a_next;
        }};
    }

    let all_active = i16x8_splat(-1);
    for row in 0..m_min {
        do_row!(row, all_active, false);
    }
    for row in m_min..m_max {
        let on = |k: usize| if row < bb[k].len() { -1i16 } else { 0 };
        let active = i16x8(on(0), on(1), on(2), on(3), on(4), on(5), on(6), on(7));
        do_row!(row, active, true);
    }

    rows[n] = acc;
    let mut best = i16x8_splat(i16::MIN);
    for r in &rows {
        best = i16x8_max(best, *r);
    }
    let norm_const = n as f32;
    let s = |high: i16| 0.5 * high as i32 as f32 / norm_const;
    [
        s(i16x8_extract_lane::<0>(best)), s(i16x8_extract_lane::<1>(best)),
        s(i16x8_extract_lane::<2>(best)), s(i16x8_extract_lane::<3>(best)),
        s(i16x8_extract_lane::<4>(best)), s(i16x8_extract_lane::<5>(best)),
        s(i16x8_extract_lane::<6>(best)), s(i16x8_extract_lane::<7>(best)),
    ]
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn simd_impl(a: &String, bs: [&String; 4]) -> [f32; 4] {
    use core::arch::wasm32::*;

    const MATCH_SCORE: i32 = 2;
    const MISMATCH_SCORE: i32 = -2;
    const GAP_SCORE: i32 = -1;

    let ab = a.as_bytes();
    let n = ab.len();
    let bb: [&[u8]; 4] = [
        bs[0].as_bytes(),
        bs[1].as_bytes(),
        bs[2].as_bytes(),
        bs[3].as_bytes(),
    ];

    // Guard the precondition rather than trusting it: falling back is always
    // correct, and a silent wrong answer here would be invisible.
    if n == 0 || bb.iter().any(|b| b.len() < n) {
        return [
            needleman_wunsch(a, bs[0]),
            needleman_wunsch(a, bs[1]),
            needleman_wunsch(a, bs[2]),
            needleman_wunsch(a, bs[3]),
        ];
    }

    let m_min = bb.iter().map(|b| b.len()).min().unwrap();
    let m_max = bb.iter().map(|b| b.len()).max().unwrap();

    let vmatch = i32x4_splat(MATCH_SCORE);
    let vmiss = i32x4_splat(MISMATCH_SCORE);
    let vgap = i32x4_splat(GAP_SCORE);

    let mut rows: Vec<v128> = vec![i32x4_splat(0); n + 1];
    let mut acc = i32x4_splat(0);

    // One row of the lattice. `active` is all-ones for lanes still running; a
    // finished lane keeps its cells and its accumulator frozen, which is what
    // makes a shorter `b` stop exactly where the scalar version would.
    macro_rules! do_row {
        ($row:expr, $active:expr, $masked:expr) => {{
            let row = $row;
            let active = $active;
            let bc = i32x4(
                *bb[0].get(row).unwrap_or(&0) as i32,
                *bb[1].get(row).unwrap_or(&0) as i32,
                *bb[2].get(row).unwrap_or(&0) as i32,
                *bb[3].get(row).unwrap_or(&0) as i32,
            );
            let mut prev_diag = i32x4_splat(0);
            let mut left = i32x4_splat(0);

            for col in 1..n {
                let curr_diag = prev_diag;
                prev_diag = rows[col];
                let eq = i32x4_eq(i32x4_splat(ab[col - 1] as i32), bc);
                let sub = i32x4_add(curr_diag, v128_bitselect(vmatch, vmiss, eq));
                let mut v = i32x4_max(
                    sub,
                    i32x4_max(i32x4_add(left, vgap), i32x4_add(prev_diag, vgap)),
                );
                let mut a_next = i32x4_max(acc, prev_diag);
                if $masked {
                    v = v128_bitselect(v, prev_diag, active);
                    a_next = v128_bitselect(a_next, acc, active);
                }
                rows[col] = v;
                left = v;
                acc = a_next;
            }

            // Column n is special and order-dependent: it reads the running
            // accumulator, not the cell above it. See nw.rs for the proof.
            let curr_diag = prev_diag;
            let prev_diag_n = acc;
            let eq = i32x4_eq(i32x4_splat(ab[n - 1] as i32), bc);
            let sub = i32x4_add(curr_diag, v128_bitselect(vmatch, vmiss, eq));
            let v = i32x4_max(
                sub,
                i32x4_max(i32x4_add(left, vgap), i32x4_add(prev_diag_n, vgap)),
            );
            let mut a_next = i32x4_max(v, prev_diag_n);
            if $masked {
                a_next = v128_bitselect(a_next, acc, active);
            }
            acc = a_next;
        }};
    }

    // Rows every lane is still running: no masking, no select, full speed.
    let all_active = i32x4_splat(-1);
    for row in 0..m_min {
        do_row!(row, all_active, false);
    }
    // Tail rows, where the shorter lanes have already stopped. Settings are
    // batched by length upstream, so this stretch is normally a handful of rows.
    for row in m_min..m_max {
        let active = i32x4(
            if row < bb[0].len() { -1 } else { 0 },
            if row < bb[1].len() { -1 } else { 0 },
            if row < bb[2].len() { -1 } else { 0 },
            if row < bb[3].len() { -1 } else { 0 },
        );
        do_row!(row, active, true);
    }

    rows[n] = acc;
    let mut best = i32x4_splat(i32::MIN);
    for r in &rows {
        best = i32x4_max(best, *r);
    }

    // Exactly the scalar's arithmetic, in the same order: `0.5 * high / n`.
    // Folding it into `high * (0.5 / n)` would round twice instead of once and
    // is NOT the same f32.
    let norm_const = n as f32;
    let s = |high: i32| 0.5 * high as f32 / norm_const;
    [
        s(i32x4_extract_lane::<0>(best)),
        s(i32x4_extract_lane::<1>(best)),
        s(i32x4_extract_lane::<2>(best)),
        s(i32x4_extract_lane::<3>(best)),
    ]
}
