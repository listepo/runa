//! P5.3 sampling kernels (plan D23): C ABI, `@Vector` SIMD softmax.
const std = @import("std");
const math = std.math;

const VLEN = 8;
const V = @Vector(VLEN, f32);

const Item = struct {
    p: f32,
    i: u32,
};

fn load_v(s: []const f32) V {
    var tmp: [VLEN]f32 = undefined;
    @memcpy(&tmp, s[0..VLEN]);
    return tmp;
}

fn store_v(s: []f32, v: V) void {
    const tmp: [VLEN]f32 = v;
    @memcpy(s[0..VLEN], &tmp);
}

export fn runa_softmax_f32_scalar(in_ptr: [*]const f32, out_ptr: [*]f32, n: usize) void {
    if (n == 0) return;
    const in_s = in_ptr[0..n];
    const out_s = out_ptr[0..n];

    var max_v: f32 = -math.inf(f32);
    var i: usize = 0;
    if (n >= VLEN) {
        var vmax: V = @splat(-math.inf(f32));
        while (i + VLEN <= n) : (i += VLEN) {
            vmax = @max(vmax, load_v(in_s[i..]));
        }
        max_v = @reduce(.Max, vmax);
    }
    while (i < n) : (i += 1) {
        max_v = @max(max_v, in_s[i]);
    }

    var sum: f32 = 0;
    i = 0;
    while (i < n) : (i += 1) {
        const e = @exp(in_s[i] - max_v);
        out_s[i] = e;
        sum += e;
    }
    if (sum <= 0) return;
    const inv = 1.0 / sum;
    const vinv: V = @splat(inv);
    i = 0;
    while (i + VLEN <= n) : (i += VLEN) {
        store_v(out_s[i..], load_v(out_s[i..]) * vinv);
    }
    while (i < n) : (i += 1) {
        out_s[i] *= inv;
    }
}

fn renormalize(p: []f32) void {
    var sum: f32 = 0;
    for (p) |x| sum += x;
    if (sum <= 0) return;
    const inv = 1.0 / sum;
    for (p) |*x| x.* *= inv;
}

fn less_desc(_: void, a: Item, b: Item) bool {
    return a.p > b.p;
}

fn scratch(n: usize) ?[]Item {
    return std.heap.page_allocator.alloc(Item, n) catch null;
}

export fn runa_top_k_f32(p_ptr: [*]f32, n: usize, k: i32) void {
    if (n == 0 or k <= 0 or @as(usize, @intCast(k)) >= n) return;
    const p = p_ptr[0..n];
    const items = scratch(n) orelse return;
    defer std.heap.page_allocator.free(items);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        items[i] = .{ .p = p[i], .i = @intCast(i) };
    }
    std.mem.sort(Item, items, {}, less_desc);
    @memset(p, 0);
    const kk: usize = @intCast(k);
    i = 0;
    while (i < kk) : (i += 1) {
        p[items[i].i] = items[i].p;
    }
    renormalize(p);
}

export fn runa_top_p_f32(p_ptr: [*]f32, n: usize, top_p: f32) void {
    if (n == 0 or top_p >= 1.0) return;
    const p = p_ptr[0..n];
    if (top_p <= 0) {
        var best: usize = 0;
        var i: usize = 1;
        while (i < n) : (i += 1) {
            if (p[i] > p[best]) best = i;
        }
        @memset(p, 0);
        p[best] = 1.0;
        return;
    }
    const items = scratch(n) orelse return;
    defer std.heap.page_allocator.free(items);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        items[i] = .{ .p = p[i], .i = @intCast(i) };
    }
    std.mem.sort(Item, items, {}, less_desc);
    var cum: f32 = 0;
    var keep: usize = 0;
    while (keep < n) : (keep += 1) {
        cum += items[keep].p;
        if (cum >= top_p) {
            keep += 1;
            break;
        }
    }
    @memset(p, 0);
    i = 0;
    while (i < keep) : (i += 1) {
        p[items[i].i] = items[i].p;
    }
    renormalize(p);
}

export fn runa_min_p_f32(p_ptr: [*]f32, n: usize, min_p: f32) void {
    if (n == 0 or min_p <= 0) return;
    const p = p_ptr[0..n];
    var mx = p[0];
    var i: usize = 1;
    while (i < n) : (i += 1) {
        mx = @max(mx, p[i]);
    }
    const thr = min_p * mx;
    i = 0;
    while (i < n) : (i += 1) {
        if (p[i] < thr) p[i] = 0;
    }
    renormalize(p);
}

fn xorshift32(s: *u32) u32 {
    var x = s.*;
    if (x == 0) x = 1;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s.* = x;
    return x;
}

export fn runa_sample_token_f32(
    logits: [*]const f32,
    n: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    seed: *u32,
) i32 {
    if (n == 0) return 0;
    const src = logits[0..n];
    if (temperature <= 0) {
        var best: usize = 0;
        var i: usize = 1;
        while (i < n) : (i += 1) {
            if (src[i] > src[best]) best = i;
        }
        return @intCast(best);
    }
    const buf = std.heap.page_allocator.alloc(f32, n) catch return 0;
    defer std.heap.page_allocator.free(buf);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        buf[i] = src[i] / temperature;
    }
    runa_softmax_f32_scalar(buf.ptr, buf.ptr, n);
    runa_top_k_f32(buf.ptr, n, top_k);
    runa_top_p_f32(buf.ptr, n, top_p);
    runa_min_p_f32(buf.ptr, n, min_p);

    const u: f32 = @as(f32, @floatFromInt(xorshift32(seed) >> 8)) / @as(f32, @floatFromInt(@as(u32, 1) << 24));
    var cum: f32 = 0;
    var pick: i32 = @intCast(n - 1);
    i = 0;
    while (i < n) : (i += 1) {
        cum += buf[i];
        if (u <= cum) {
            pick = @intCast(i);
            break;
        }
    }
    return pick;
}
