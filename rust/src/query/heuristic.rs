use crate::ff_config;
use crate::index::schema::*;
use crate::index::TuneIndex;
use aho_corasick::{AhoCorasick, Match};
use std::collections::HashMap;

pub struct ScoredName {
    pub tune_id: TuneID,
    pub alias_index: usize,
    pub ngram_score: f32,
}

/// Scores every setting against the query's n-grams and returns the best
/// `top_n` of them, best first. Only the first `top_n` are meaningful — the
/// caller never looked at the rest.
pub fn run_transcription_query<'a>(
    query: &String,
    // settings_feats: &SettingsFeats,
    tune_index: &'a TuneIndex,
    top_n: usize,
) -> Vec<(&'a SettingID, usize)> {
    // The n-grams are a FIXED width (QUERY_NGRAM_SIZE_CONTOUR = 4), so an
    // Aho-Corasick automaton is more machinery than the problem needs: four
    // bytes pack into a u32 and a match is an equality test on that word. What
    // the automaton actually did, we reproduce exactly:
    //
    //   * `ngrams_str` keeps DUPLICATES, and aho-corasick treats each duplicate
    //     as its own pattern id, so `find_overlapping_iter` reported one match
    //     per duplicate. The score was therefore
    //         sum over distinct grams g of  (times g occurs in the QUERY)
    //                                     * (times g occurs in the SETTING).
    //     `mult` below holds the first factor, so accumulating it per hit gives
    //     the identical total.
    //   * Overlapping matches means every window position counts, which is
    //     exactly what rolling the code one byte at a time visits.
    //
    // `filter` is a fast reject only — a Bloom-style bit per hashed code. False
    // positives are possible and harmless: they fall through to the exact
    // lookup in `mult`, which returns nothing for a non-member. So this cannot
    // change the score, only the time spent finding it. Working on raw bytes
    // rather than the 48-letter contour alphabet keeps it correct whatever the
    // input happens to contain.
    const N: usize = ff_config::QUERY_NGRAM_SIZE_CONTOUR;
    const FILTER_BITS: u32 = 21; // 2^21 bits = 256 KiB, cleared once per query
    let filter_len: usize = 1 << (FILTER_BITS - 6);
    let mut filter: Vec<u64> = vec![0; filter_len];
    let hash = |code: u32| -> usize {
        ((code.wrapping_mul(0x9E37_79B1) >> (32 - FILTER_BITS)) as usize) & ((1 << FILTER_BITS) - 1)
    };

    let qb = query.as_bytes();
    let mut mult: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    if qb.len() >= N {
        for i in 0..=qb.len() - N {
            let mut code: u32 = 0;
            for k in 0..N {
                code = (code << 8) | qb[i + k] as u32;
            }
            *mult.entry(code).or_insert(0) += 1;
            let h = hash(code);
            filter[h >> 6] |= 1u64 << (h & 63);
        }
    }
    // ngrams_str's short-query branch returns the whole string as one "n-gram";
    // with fewer than N characters there is no window to match, and
    // aho-corasick would only ever match it against a setting containing that
    // exact short string as a substring. Reproduce that case through the old
    // path rather than guessing.
    let short_query_fallback = qb.len() < N;
    let ac_short = if short_query_fallback {
        Some(AhoCorasick::new_auto_configured(&ngrams_str(
            query,
            ff_config::QUERY_NGRAM_SIZE_CONTOUR,
        )))
    } else {
        None
    };

    // Straight into a Vec, borrowing the ids: the old code built a HashMap only
    // to iterate it once and collect it back out (55k hash inserts per query
    // for nothing) and cloned every id (55k String allocations per query, for
    // ids that outlive the call inside the index). And `.count()` rather than
    // `.collect::<Vec<Match>>().len()`: the old form allocated a Vec of every
    // match, per setting, purely to read its length.
    let mut ranked: Vec<(&'a SettingID, usize)> = Vec::with_capacity(tune_index.settings.len());
    for (setting_id, setting) in &tune_index.settings {
        let score = if let Some(ac) = &ac_short {
            ac.find_overlapping_iter(&setting.contour).count()
        } else {
            let sb = setting.contour.as_bytes();
            if sb.len() < N {
                0
            } else {
                let mut code: u32 = 0;
                for k in 0..N - 1 {
                    code = (code << 8) | sb[k] as u32;
                }
                let mut total: usize = 0;
                for k in N - 1..sb.len() {
                    code = (code << 8) | sb[k] as u32;
                    let h = hash(code);
                    if filter[h >> 6] & (1u64 << (h & 63)) != 0 {
                        if let Some(m) = mult.get(&code) {
                            total += *m as usize;
                        }
                    }
                }
                total
            }
        };
        ranked.push((setting_id, score));
    }

    // Ties broken by setting id, deliberately. `tune_index.settings` is a
    // HashMap, whose iteration order differs between instances, and `sort_by`
    // is stable — so without this the order among equal match counts was drawn
    // fresh on every query. With 55k settings and long plateaus of equal
    // counts, that made the QUERY_REPASS_SIZE cut-off arbitrary and the engine
    // non-reproducible: the same query returned a different top-1 on ~1 window
    // in 80, and a different top-10 on ~1 in 6. Lexicographic, not numeric —
    // any total order will do, this one is free.
    let cmp = |x: &(&SettingID, usize), y: &(&SettingID, usize)| y.1.cmp(&x.1).then_with(|| x.0.cmp(y.0));

    // Select-then-sort rather than sorting all 55k to keep 2000. Because the
    // comparator above is a STRICT TOTAL ORDER (ids are unique, so no two
    // elements compare Equal), the set of the `top_n` greatest is uniquely
    // determined — partial selection returns exactly what a full sort would
    // have put in those positions. Not an approximation.
    if ranked.len() > top_n {
        ranked.select_nth_unstable_by(top_n, cmp);
        ranked.truncate(top_n);
    }
    ranked.sort_unstable_by(cmp);
    ranked
}

/// The n-gram pass of a pitch-class query ('a'..'l', index folded alike) in all twelve keys at
/// once (Cadence, 2026-11-07). Every query 4-gram is moved down r = 0..11 semitones; a setting's
/// gram equal to a gram moved down r counts for the transposition r (the query is the setting r
/// semitones up). The pass over the index is then run_transcription_query's own (rolling 4-byte
/// code, Bloom filter, lookup on hits), and scores the twelve transpositions of every setting
/// at once; r = 0 is exactly run_transcription_query's score.
///
/// Returns run_transcription_query's own top_n (r = 0, same order — the written key's
/// shortlist is untouched), then the best `extra` (setting, r != 0) pairs, best first (ties:
/// setting id, then r). Falls back to the written key alone when the query has fewer than 4
/// symbols or any outside 'a'..'l'.
pub fn run_transcription_query_any_key<'a>(
    query: &String,
    tune_index: &'a TuneIndex,
    top_n: usize,
    extra: usize,
) -> Vec<(&'a SettingID, u8)> {
    let q: Vec<u8> = query.bytes().map(|b| b.wrapping_sub(b'a')).collect();
    if q.len() < ff_config::QUERY_NGRAM_SIZE_CONTOUR || q.iter().any(|&c| c >= 12) {
        return run_transcription_query(query, tune_index, top_n).into_iter().map(|(id, _)| (id, 0)).collect();
    }
    // Every query gram moved down r = 0..11 semitones, as the 4-byte code a setting's gram has
    // when it matches the query r semitones up; per code, the (r, count) it scores.
    let mut by_code: HashMap<u32, [u32; 12]> = HashMap::new();
    for i in 0..=q.len() - 4 {
        for r in 0..12u8 {
            let mut code: u32 = 0;
            for k in 0..4 {
                code = (code << 8) | (b'a' + (q[i + k] + 12 - r) % 12) as u32;
            }
            by_code.entry(code).or_insert([0; 12])[r as usize] += 1;
        }
    }
    // A gram of four pitch classes packed four bits a class: one of 2^16 codes, rolled with a
    // shift, an or and a mask — no arithmetic chained from one symbol to the next, which a
    // base-12 index costs (its multiply-and-modulo alone took twice the written key's whole
    // pass over the 12.5 million symbols of the index). `present` (8 KB, one bit per code,
    // stays in the fastest cache) says whether the query has it in some key; `slot` then gives
    // the range of its (r, count) entries (2026-11-08: it replaced a Bloom filter and a hashed
    // lookup). The entries are the same, so the scores are.
    const CODES: usize = 1 << 16;
    let pack = |code: u32| -> usize {
        let b = code.to_be_bytes();
        ((b[0] - b'a') as usize) << 12 | ((b[1] - b'a') as usize) << 8 | ((b[2] - b'a') as usize) << 4 | (b[3] - b'a') as usize
    };
    let mut present: Vec<u64> = vec![0; CODES / 64];
    let mut slot: Vec<u16> = vec![0; CODES];
    let mut ranges: Vec<(u32, u32)> = vec![(0, 0)];
    let mut entries: Vec<(u8, u32)> = Vec::new();
    for (code, per_r) in &by_code {
        let from = entries.len() as u32;
        for r in 0..12u8 {
            if per_r[r as usize] > 0 {
                entries.push((r, per_r[r as usize]));
            }
        }
        let g = pack(*code);
        slot[g] = ranges.len() as u16;
        ranges.push((from, entries.len() as u32));
        present[g >> 6] |= 1u64 << (g & 63);
    }

    // The other keys' scores, every setting's, and how many (setting, r != 0) pairs reach each
    // score: the best `extra` pairs are then everything above the score where the count
    // reaches `extra`, and the best of the ties at it by (id, r) — the very set and order a
    // bounded heap kept (score down, then id, then r), without its comparisons, most of
    // them between ids of equal scores (2026-11-08).
    let mut other: Vec<[u32; 11]> = Vec::with_capacity(if extra > 0 { tune_index.grams.ids.len() } else { 0 });
    let mut count_at: Vec<u32> = Vec::new();

    // The settings' grams, computed when the index was folded (PitchClassGrams). NO_GRAM's bit
    // is never set: no query gram is it.
    let grams = &tune_index.grams;
    let mut written: Vec<(&'a SettingID, usize)> = Vec::with_capacity(grams.ids.len());
    for (i, setting_id) in grams.ids.iter().enumerate() {
        let mut sc = [0usize; 12];
        for &code in &grams.codes[grams.starts[i] as usize..grams.starts[i + 1] as usize] {
            let code = code as usize;
            if present[code >> 6] & (1u64 << (code & 63)) != 0 {
                let (from, to) = ranges[slot[code] as usize];
                for &(r, m) in &entries[from as usize..to as usize] {
                    sc[r as usize] += m as usize;
                }
            }
        }
        written.push((setting_id, sc[0]));
        if extra > 0 {
            let mut o = [0u32; 11];
            for r in 1..12 {
                let v = sc[r];
                o[r - 1] = v as u32;
                if v > 0 {
                    if v >= count_at.len() {
                        count_at.resize(v + 1, 0);
                    }
                    count_at[v] += 1;
                }
            }
            other.push(o);
        }
    }
    // The score of the extra-th best pair (`floor`) and how many of the pairs AT it are kept.
    let mut floor = 1usize;
    let mut at_floor = usize::MAX;
    let mut above = 0usize;
    for v in (1..count_at.len()).rev() {
        let c = count_at[v] as usize;
        if above + c >= extra {
            floor = v;
            at_floor = extra - above;
            break;
        }
        above += c;
    }
    let mut kept: Vec<(usize, &'a SettingID, u8)> = Vec::new();
    let mut ties: Vec<(&'a SettingID, u8)> = Vec::new();
    if extra > 0 {
        for (i, o) in other.iter().enumerate() {
            for (k, &v) in o.iter().enumerate() {
                let v = v as usize;
                if v > floor {
                    kept.push((v, written[i].0, k as u8 + 1));
                } else if v == floor {
                    ties.push((written[i].0, k as u8 + 1));
                }
            }
        }
        // the ties kept: smallest id, then smallest r
        if ties.len() > at_floor {
            ties.select_nth_unstable_by(at_floor, |x, y| x.0.cmp(y.0).then_with(|| x.1.cmp(&y.1)));
            ties.truncate(at_floor);
        }
        kept.extend(ties.into_iter().map(|(id, r)| (floor, id, r)));
        kept.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(y.1)).then_with(|| x.2.cmp(&y.2)));
    }
    // the written key exactly as run_transcription_query ranks it
    let cmp = |x: &(&SettingID, usize), y: &(&SettingID, usize)| y.1.cmp(&x.1).then_with(|| x.0.cmp(y.0));
    if written.len() > top_n {
        written.select_nth_unstable_by(top_n, cmp);
        written.truncate(top_n);
    }
    written.sort_unstable_by(cmp);
    let mut out: Vec<(&'a SettingID, u8)> = written.into_iter().map(|(id, _)| (id, 0)).collect();
    out.extend(kept.into_iter().map(|(_, id, r)| (id, r)));
    out
}

pub fn run_name_query(query: &String, tune_index: &TuneIndex) -> Vec<ScoredName> {
    let query = query.to_lowercase();
    let query_len = query.len() as f32;
    let ngrams = ngrams_str(&query, ff_config::QUERY_NGRAM_SIZE_NAME);

    let mut scored_names: Vec<ScoredName> = Vec::new();
    let ac = AhoCorasick::new_auto_configured(&ngrams);
    for (tune_id, aliases) in &tune_index.aliases {
        for (alias_id, alias) in aliases.iter().enumerate() {
            let score = ac
                .find_overlapping_iter(&alias)
                .collect::<Vec<Match>>()
                .len();
            let score = score as f32 / f32::max((alias.len() as f32).sqrt(), query_len);
            scored_names.push(ScoredName {
                tune_id: tune_id.clone(),
                alias_index: alias_id,
                ngram_score: score,
            });
        }
    }

    return scored_names;
}

pub fn ngrams_str(query: &String, n: usize) -> Vec<String> {
    // n=2 -> 'bigram'
    // n=3 -> 'trigram'
    //  etc...
    let mut grams: Vec<String> = Vec::new();
    let chars: Vec<char> = query.chars().collect();
    if chars.len() < n {
        grams.push(chars.iter().collect());
        return grams;
    }

    for i in 0..chars.len() - (n - 1) {
        let ngram: String = chars[i..i + n].iter().collect();
        grams.push(ngram);
    }
    return grams;
}
