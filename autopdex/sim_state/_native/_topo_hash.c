/* autopdex/_topo_hash.c
 *
 * Native O(n) topology deduplication / lookup for AutoPDEx mesh building.
 *
 * Two functions, both using an open-addressing hash table with simple linear
 * probing and a SplitMix64 hash. Rows are assumed canonical (sorted per row by
 * the Python caller), so two rows are equal iff their uint32 words are equal.
 *
 * Compiled via cffi (out-of-line API mode); see _topo_hash_cffi.py.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t splitmix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* Hash a canonical row of `width` uint32 words into 64 bits. width is 2, 3 or 4.
 * width==2 packs (a,b) into a single uint64; otherwise the words are folded. */
static inline uint64_t hash_row(const uint32_t *r, int width) {
    if (width == 2) {
        uint64_t k = ((uint64_t)r[0] << 32) | (uint64_t)r[1];
        return splitmix64(k);
    }
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int j = 0; j < width; ++j) {
        h = splitmix64(h ^ (uint64_t)r[j]);
    }
    return h;
}

static inline int row_eq(const uint32_t *a, const uint32_t *b, int width) {
    for (int j = 0; j < width; ++j) {
        if (a[j] != b[j]) return 0;
    }
    return 1;
}

/* Smallest power-of-two capacity with load factor < 0.75, i.e. n/cap < 3/4. */
static inline int64_t capacity_for(int64_t n) {
    int64_t cap = 8;
    while (cap * 3 <= n * 4) cap <<= 1;
    return cap;
}

/* Pass 2: deduplicate all edge/facet occurrences.
 *   rows     : (n, width) uint32, C-contiguous, canonicalized
 *   out_inv  : [n]        out_inv[i] = entity id of rows[i] (first-occurrence ids)
 *   out_uniq : [n*width]  unique rows in first-occurrence order (first n_unique used)
 * Returns n_unique (>= 0) or a negative value on allocation failure. */
int64_t factorize_entity_rows(const uint32_t *rows, int64_t n, int width,
                              int64_t *out_inv, uint32_t *out_uniq) {
    if (n <= 0) return 0;

    int64_t cap = capacity_for(n);
    uint64_t mask = (uint64_t)(cap - 1);

    uint32_t *keys = (uint32_t *)malloc((size_t)cap * (size_t)width * sizeof(uint32_t));
    int64_t  *vals = (int64_t  *)malloc((size_t)cap * sizeof(int64_t));
    uint8_t  *occ  = (uint8_t  *)calloc((size_t)cap, sizeof(uint8_t));
    if (!keys || !vals || !occ) {
        free(keys); free(vals); free(occ);
        return -1;
    }

    int64_t n_unique = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t *r = rows + i * (int64_t)width;
        uint64_t h = hash_row(r, width) & mask;
        int64_t id;
        for (;;) {
            if (!occ[h]) {
                occ[h] = 1;
                memcpy(keys + (int64_t)h * width, r, (size_t)width * sizeof(uint32_t));
                id = n_unique;
                vals[h] = id;
                memcpy(out_uniq + n_unique * width, r, (size_t)width * sizeof(uint32_t));
                ++n_unique;
                break;
            }
            if (row_eq(keys + (int64_t)h * width, r, width)) {
                id = vals[h];
                break;
            }
            h = (h + 1) & mask;
        }
        out_inv[i] = id;
    }

    free(keys); free(vals); free(occ);
    return n_unique;
}

/* --- uint64-row deduplication (used for exact mesh-point merging) ---------- */

static inline uint64_t hash_row_u64(const uint64_t *r, int width) {
    if (width == 1) return splitmix64(r[0]);
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int j = 0; j < width; ++j) {
        h = splitmix64(h ^ r[j]);
    }
    return h;
}

static inline int row_eq_u64(const uint64_t *a, const uint64_t *b, int width) {
    for (int j = 0; j < width; ++j) {
        if (a[j] != b[j]) return 0;
    }
    return 1;
}

/* Deduplicate exact uint64 rows (e.g. float64 point coordinates viewed as bits).
 *   rows      : (n, width) uint64, C-contiguous, width 1..4 (not canonicalized)
 *   out_inv   : [n]        out_inv[i] = unique id of rows[i] (first-occurrence)
 *   out_first : [n]        out_first[id] = first row index that produced id
 * Returns n_unique (>= 0) or a negative value on allocation failure. */
int64_t factorize_u64_rows(const uint64_t *rows, int64_t n, int width,
                           int64_t *out_inv, int64_t *out_first) {
    if (n <= 0) return 0;

    int64_t cap = capacity_for(n);
    uint64_t mask = (uint64_t)(cap - 1);

    uint64_t *keys = (uint64_t *)malloc((size_t)cap * (size_t)width * sizeof(uint64_t));
    int64_t  *vals = (int64_t  *)malloc((size_t)cap * sizeof(int64_t));
    uint8_t  *occ  = (uint8_t  *)calloc((size_t)cap, sizeof(uint8_t));
    if (!keys || !vals || !occ) {
        free(keys); free(vals); free(occ);
        return -1;
    }

    int64_t n_unique = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t *r = rows + i * (int64_t)width;
        uint64_t h = hash_row_u64(r, width) & mask;
        int64_t id;
        for (;;) {
            if (!occ[h]) {
                occ[h] = 1;
                memcpy(keys + (int64_t)h * width, r, (size_t)width * sizeof(uint64_t));
                id = n_unique;
                vals[h] = id;
                out_first[n_unique] = i;
                ++n_unique;
                break;
            }
            if (row_eq_u64(keys + (int64_t)h * width, r, width)) {
                id = vals[h];
                break;
            }
            h = (h + 1) & mask;
        }
        out_inv[i] = id;
    }

    free(keys); free(vals); free(occ);
    return n_unique;
}

/* Pass 3: look up query rows against a known entity table.
 *   entity_rows : (n_entities, width) uint32, canonicalized
 *   query_rows  : (n_queries,  width) uint32, canonicalized
 *   out_ids     : [n_queries]  entity index (row in entity_rows) or -1 if absent
 * Returns the number of not-found queries, or negative on allocation failure. */
int64_t lookup_entity_rows(const uint32_t *entity_rows, int64_t n_entities, int width,
                           const uint32_t *query_rows, int64_t n_queries,
                           int64_t *out_ids) {
    int64_t cap = capacity_for(n_entities);
    uint64_t mask = (uint64_t)(cap - 1);

    uint32_t *keys = (uint32_t *)malloc((size_t)cap * (size_t)width * sizeof(uint32_t));
    int64_t  *vals = (int64_t  *)malloc((size_t)cap * sizeof(int64_t));
    uint8_t  *occ  = (uint8_t  *)calloc((size_t)cap, sizeof(uint8_t));
    if (!keys || !vals || !occ) {
        free(keys); free(vals); free(occ);
        return -1;
    }

    for (int64_t i = 0; i < n_entities; ++i) {
        const uint32_t *r = entity_rows + i * (int64_t)width;
        uint64_t h = hash_row(r, width) & mask;
        for (;;) {
            if (!occ[h]) {
                occ[h] = 1;
                memcpy(keys + (int64_t)h * width, r, (size_t)width * sizeof(uint32_t));
                vals[h] = i;
                break;
            }
            if (row_eq(keys + (int64_t)h * width, r, width)) {
                break; /* duplicate entity row: keep first id */
            }
            h = (h + 1) & mask;
        }
    }

    int64_t misses = 0;
    for (int64_t q = 0; q < n_queries; ++q) {
        const uint32_t *r = query_rows + q * (int64_t)width;
        uint64_t h = hash_row(r, width) & mask;
        int64_t id = -1;
        for (;;) {
            if (!occ[h]) break;
            if (row_eq(keys + (int64_t)h * width, r, width)) {
                id = vals[h];
                break;
            }
            h = (h + 1) & mask;
        }
        out_ids[q] = id;
        if (id < 0) ++misses;
    }

    free(keys); free(vals); free(occ);
    return misses;
}
