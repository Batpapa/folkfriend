use std::cmp;

const MATCH_SCORE: i32 = 2;
const MISMATCH_SCORE: i32 = -2;
const GAP_SCORE: i32 = -1;

pub fn needleman_wunsch(a: &String, b: &String) -> f32 {

    //  Memory-efficient version of Needleman-Wunsch written for Rust.
    //    ~ Tom Wyllie 2021
    //
    //  Rewritten for speed (Cadence, 2026-09-01) — BIT-IDENTICAL output, this
    //  is the hottest loop in the engine (~80% of a query's cost, run
    //  QUERY_REPASS_SIZE times per query). Three changes, all mechanical:
    //
    //   1. Byte comparison instead of &str slicing. The contour alphabet is 48
    //      ASCII letters (ff_config::CONTOUR_TO_QUERY_CHAR), so one byte is
    //      exactly one note.
    //   2. `b`'s character is hoisted out of the inner loop — it only depends
    //      on the row.
    //   3. `last_row[last_col_index]` is kept in a register (`acc`) instead of
    //      being re-loaded and re-stored on every single cell. That store also
    //      aliased `last_row[col]` as far as the compiler could tell, which
    //      forced a reload of the neighbours on every iteration.
    //
    //  Why (3) is exact, since it is the subtle one. The original body is:
    //
    //      curr_diag = prev_diag;
    //      prev_diag = last_row[col];                    // previous row's cell
    //      last_row[col] = max(curr_diag + s,
    //                          max(last_row[col-1] + GAP, last_row[col] + GAP));
    //      last_row[N]   = max(last_row[N], prev_diag);  // runs on EVERY col
    //
    //  `last_row[col]` on the right-hand side is read before the write, so it
    //  is exactly `prev_diag` — including at col == N, where `prev_diag` picks
    //  up whatever the accumulator line left there earlier in this same row.
    //  So the recurrence never needs to read the array for its own cell, and
    //  the only cell that must stay in sync with `acc` is N, which nothing
    //  else reads (`last_row[col-1]` tops out at N-1). Handling col == N once,
    //  after the loop, therefore reproduces the original exactly — including
    //  its order dependence, which is why this accumulator must NOT simply be
    //  hoisted to the end of the row.

    if a.len() == 0 || b.len() == 0 {
        return 0.0;
    }

    // Swap a and b such that b always longer than a
    let (a, b) = if a.len() > b.len() { (&b, &a) } else { (&a, &b) };

    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let n = a_bytes.len();

    let mut last_row: Vec<i32> = vec![0; n + 1];

    //       a1 a2 a3 .. aN
    //    b1
    //    b2
    //    b3
    //    ..
    //    bN

    // Mirrors last_row[n] for the whole scan; written back once at the end.
    let mut acc: i32 = 0;

    //  Populate dynamic programming lattice
    for row in 0..b_bytes.len() {
        let bc = b_bytes[row];
        let mut prev_diag: i32 = 0;
        // last_row[0] is never written, so the cell to the left of column 1 is
        // always 0 — same as the original.
        let mut left: i32 = 0;

        // Columns 1..n as a zip of two equal-length slices rather than indexed
        // accesses: `a_bytes[col-1]` and `last_row[col]` each cost a bounds
        // check per lattice cell otherwise, and there are n*m cells. Same
        // arithmetic, same order — no `unsafe` needed to drop the checks.
        for (&ac, cell) in a_bytes[..n - 1].iter().zip(last_row[1..n].iter_mut()) {
            let curr_diag = prev_diag;
            prev_diag = *cell;
            let sub = curr_diag + if ac == bc { MATCH_SCORE } else { MISMATCH_SCORE };
            let v = cmp::max(sub, cmp::max(left + GAP_SCORE, prev_diag + GAP_SCORE));
            *cell = v;
            left = v;
            acc = cmp::max(acc, prev_diag);
        }

        // col == n, where the cell being written IS the accumulator.
        let curr_diag = prev_diag;
        prev_diag = acc;
        let sub = curr_diag + if a_bytes[n - 1] == bc { MATCH_SCORE } else { MISMATCH_SCORE };
        let cell = cmp::max(sub, cmp::max(left + GAP_SCORE, prev_diag + GAP_SCORE));
        acc = cmp::max(cell, prev_diag);
    }

    last_row[n] = acc;

    //  Normalised [0, 1].
    let highscore: f32 = *last_row.iter().max().unwrap_or(&0) as f32;
    let norm_const: f32 = n as f32;
    return 0.5 * highscore / norm_const;
}
