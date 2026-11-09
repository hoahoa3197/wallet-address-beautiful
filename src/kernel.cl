// secp256k1 point stepping + keccak256 + pattern match.
// Field elements: 8 x 32-bit limbs, little-endian limb order (d[0] least significant).
// M (points per round, one shared inversion) is set by the host via -D M=...

typedef struct { uint d[8]; } fe;

// r + carry*2^256 -> reduced mod p, input < 2p. Uses 2^256 - p = 0x1000003D1.
inline fe fe_reduce_once(fe r, ulong carry) {
    fe t;
    ulong c = (ulong)r.d[0] + 0x3D1u;
    t.d[0] = (uint)c; c >>= 32;
    c += (ulong)r.d[1] + 1u;
    t.d[1] = (uint)c; c >>= 32;
    #pragma unroll
    for (int i = 2; i < 8; i++) { c += r.d[i]; t.d[i] = (uint)c; c >>= 32; }
    return (carry | c) ? t : r;
}

inline fe fe_add(fe a, fe b) {
    fe r;
    ulong c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (ulong)a.d[i] + b.d[i]; r.d[i] = (uint)c; c >>= 32; }
    return fe_reduce_once(r, c);
}

inline fe fe_sub(fe a, fe b) {
    fe r;
    ulong br = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ulong t = (ulong)a.d[i] - b.d[i] - br;
        r.d[i] = (uint)t; br = t >> 63;
    }
    // on borrow add p == subtract 0x1000003D1 mod 2^256
    ulong t = (ulong)r.d[0] - br * 0x3D1u;
    r.d[0] = (uint)t; ulong b2 = t >> 63;
    t = (ulong)r.d[1] - br - b2;
    r.d[1] = (uint)t; b2 = t >> 63;
    #pragma unroll
    for (int i = 2; i < 8; i++) { t = (ulong)r.d[i] - b2; r.d[i] = (uint)t; b2 = t >> 63; }
    return r;
}

inline fe fe_mul(fe a, fe b) {
    uint r[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) r[i] = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ulong c = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c += (ulong)a.d[i] * b.d[j] + r[i + j];
            r[i + j] = (uint)c; c >>= 32;
        }
        r[i + 8] = (uint)c;
    }
    // L + H*2^256 == L + H*977 + (H << 32)  (mod p)
    fe t;
    ulong c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c += (ulong)r[i] + (ulong)r[8 + i] * 977u;
        if (i > 0) c += r[7 + i];
        t.d[i] = (uint)c; c >>= 32;
    }
    ulong hi = c + r[15];
    c = (ulong)t.d[0] + hi * 977u; t.d[0] = (uint)c; c >>= 32;
    c += (ulong)t.d[1] + hi;       t.d[1] = (uint)c; c >>= 32;
    #pragma unroll
    for (int i = 2; i < 8; i++) { c += t.d[i]; t.d[i] = (uint)c; c >>= 32; }
    // c is 0/1; if 1 the low part is tiny, so one more fold cannot overflow
    ulong k = c;
    c = (ulong)t.d[0] + k * 0x3D1u; t.d[0] = (uint)c; c >>= 32;
    c += (ulong)t.d[1] + k;         t.d[1] = (uint)c; c >>= 32;
    #pragma unroll
    for (int i = 2; i < 8; i++) { c += t.d[i]; t.d[i] = (uint)c; c >>= 32; }
    return fe_reduce_once(t, 0);
}

inline fe fe_sqrn(fe a, int n) {
    for (int i = 0; i < n; i++) a = fe_mul(a, a);
    return a;
}

// a^(p-2), addition chain from libsecp256k1
inline fe fe_inv(fe a) {
    fe x2 = fe_mul(fe_mul(a, a), a);
    fe x3 = fe_mul(fe_mul(x2, x2), a);
    fe x6 = fe_mul(fe_sqrn(x3, 3), x3);
    fe x9 = fe_mul(fe_sqrn(x6, 3), x3);
    fe x11 = fe_mul(fe_sqrn(x9, 2), x2);
    fe x22 = fe_mul(fe_sqrn(x11, 11), x11);
    fe x44 = fe_mul(fe_sqrn(x22, 22), x22);
    fe x88 = fe_mul(fe_sqrn(x44, 44), x44);
    fe x176 = fe_mul(fe_sqrn(x88, 88), x88);
    fe x220 = fe_mul(fe_sqrn(x176, 44), x44);
    fe x223 = fe_mul(fe_sqrn(x220, 3), x3);
    fe t = fe_mul(fe_sqrn(x223, 23), x22);
    t = fe_mul(fe_sqrn(t, 5), a);
    t = fe_mul(fe_sqrn(t, 3), x2);
    return fe_mul(fe_sqrn(t, 2), a);
}

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL};
constant uint ROTC[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
constant uint PILN[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

inline void keccakf(ulong st[25]) {
    for (int round = 0; round < 24; round++) {
        ulong bc[5];
        #pragma unroll
        for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            ulong t = bc[(i + 4) % 5] ^ rotate(bc[(i + 1) % 5], 1UL);
            #pragma unroll
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        ulong t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            uint j = PILN[i];
            ulong tmp = st[j];
            st[j] = rotate(t, (ulong)ROTC[i]);
            t = tmp;
        }
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            #pragma unroll
            for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= RC[round];
    }
}

inline uint bswap32(uint v) {
    return (rotate(v, 8u) & 0x00FF00FFu) | (rotate(v, 24u) & 0xFF00FF00u);
}

inline ulong lane(uint hi_limb, uint lo_limb) {
    // big-endian bytes of two limbs, read as a little-endian u64
    return (ulong)bswap32(hi_limb) | ((ulong)bswap32(lo_limb) << 32);
}

// address = keccak256(x || y)[12..32]; word w[i] holds address bytes 4i..4i+3 (byte 4i lowest)
inline bool matches(fe x, fe y, constant uint* mask, constant uint* val) {
    ulong st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;
    st[0] = lane(x.d[7], x.d[6]); st[1] = lane(x.d[5], x.d[4]);
    st[2] = lane(x.d[3], x.d[2]); st[3] = lane(x.d[1], x.d[0]);
    st[4] = lane(y.d[7], y.d[6]); st[5] = lane(y.d[5], y.d[4]);
    st[6] = lane(y.d[3], y.d[2]); st[7] = lane(y.d[1], y.d[0]);
    st[8] = 0x01UL;                 // keccak padding after 64 bytes
    st[16] = 0x8000000000000000UL;  // last byte of the 136-byte block
    keccakf(st);
    uint w[5] = {(uint)(st[1] >> 32), (uint)st[2], (uint)(st[2] >> 32), (uint)st[3], (uint)(st[3] >> 32)};
    bool ok = true;
    #pragma unroll
    for (int i = 0; i < 5; i++) ok &= (w[i] & mask[i]) == val[i];
    return ok;
}

inline fe load_fe(constant uint* t, int j) {
    fe r;
    #pragma unroll
    for (int i = 0; i < 8; i++) r.d[i] = t[j * 8 + i];
    return r;
}

// Each work-item owns point P (key k). Per round it checks k+1 .. k+M using table tx/ty = jG (j=1..M),
// then advances P by M*G. Hit -> found = {1, gid, round*M + j}.
kernel void search(global uint* px, global uint* py, constant uint* tx, constant uint* ty,
                   constant uint* mask, constant uint* val, uint rounds, uint n, global volatile uint* found) {
    uint gid = get_global_id(0);
    fe x, y;
    #pragma unroll
    for (int i = 0; i < 8; i++) { x.d[i] = px[i * n + gid]; y.d[i] = py[i * n + gid]; }

    for (uint r = 0; r < rounds; r++) {
        fe c[M];  // prefix products of dx for batch inversion
        c[0] = fe_sub(load_fe(tx, 0), x);
        for (int j = 1; j < M; j++) c[j] = fe_mul(c[j - 1], fe_sub(load_fe(tx, j), x));
        fe inv = fe_inv(c[M - 1]);
        fe nx = x, ny = y;
        for (int j = M - 1; j >= 0; j--) {
            fe qx = load_fe(tx, j), qy = load_fe(ty, j);
            fe dx = fe_sub(qx, x);
            fe ij = j > 0 ? fe_mul(inv, c[j - 1]) : inv;
            inv = fe_mul(inv, dx);
            fe lam = fe_mul(fe_sub(qy, y), ij);
            fe x3 = fe_sub(fe_sub(fe_mul(lam, lam), x), qx);
            fe y3 = fe_sub(fe_mul(lam, fe_sub(x, x3)), y);
            if (j == M - 1) { nx = x3; ny = y3; }
            if (matches(x3, y3, mask, val) && atomic_cmpxchg(&found[0], 0u, 1u) == 0u) {
                found[1] = gid;
                found[2] = r * M + (uint)j + 1u;
            }
        }
        x = nx; y = ny;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) { px[i * n + gid] = x.d[i]; py[i * n + gid] = y.d[i]; }
}
