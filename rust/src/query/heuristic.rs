use crate::ff_config;
use crate::index::schema::*;
use crate::index::TuneIndex;
use aho_corasick::{AhoCorasick, Match};

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
