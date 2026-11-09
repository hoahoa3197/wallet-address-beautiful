use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::{BatchNormalize, Field, PrimeField};
use k256::{AffinePoint, ProjectivePoint, Scalar};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use rand_core::OsRng;
use sha3::{Digest, Keccak256};
use std::error::Error;
use std::time::Instant;
use std::{env, process};

const M: usize = 128; // keys per round per work-item (one field inversion each)
const N: usize = 1 << 15; // work-items (GPU threads)
const ROUNDS: u32 = 16; // rounds per kernel launch; keep a launch well under Windows' 2s GPU watchdog
const STRIDE_BITS: u32 = 40; // work-item g starts at k0 + g * 2^40, ranges never overlap

type Res<T> = Result<T, Box<dyn Error>>;

fn address(p: &AffinePoint) -> [u8; 20] {
    let enc = p.to_encoded_point(false);
    let h = Keccak256::digest(&enc.as_bytes()[1..]);
    h[12..].try_into().unwrap()
}

fn key_to_address(k: &Scalar) -> [u8; 20] {
    address(&(ProjectivePoint::GENERATOR * k).to_affine())
}

fn nibble(a: &[u8; 20], i: usize) -> u8 {
    if i % 2 == 0 { a[i / 2] >> 4 } else { a[i / 2] & 0xf }
}

/// (nibble position, value) pairs that must match
fn parse_pattern(prefix: &str, suffix: &str) -> Result<Vec<(usize, u8)>, String> {
    if prefix.len() + suffix.len() > 40 {
        return Err("prefix + suffix longer than 40 hex chars".into());
    }
    let hex = |c: char| c.to_digit(16).map(|d| d as u8).ok_or(format!("'{c}' is not hex (0-9, a-f)"));
    let mut pat = Vec::new();
    for (i, c) in prefix.chars().enumerate() {
        pat.push((i, hex(c)?));
    }
    let start = 40 - suffix.len();
    for (i, c) in suffix.chars().enumerate() {
        pat.push((start + i, hex(c)?));
    }
    Ok(pat)
}

fn matches(a: &[u8; 20], pat: &[(usize, u8)]) -> bool {
    pat.iter().all(|&(i, v)| nibble(a, i) == v)
}

/// EIP-55 checksummed "0x..." string
fn checksum(a: &[u8; 20]) -> String {
    let lower: String = a.iter().map(|b| format!("{b:02x}")).collect();
    let h = Keccak256::digest(lower.as_bytes());
    let mut out = String::from("0x");
    for (i, c) in lower.chars().enumerate() {
        let hn = if i % 2 == 0 { h[i / 2] >> 4 } else { h[i / 2] & 0xf };
        out.push(if hn >= 8 { c.to_ascii_uppercase() } else { c });
    }
    out
}

/// (x, y) as little-endian u32 limbs, matching the kernel's `fe`
fn limbs(p: &AffinePoint) -> ([u32; 8], [u32; 8]) {
    let enc = p.to_encoded_point(false);
    let b = enc.as_bytes();
    let f = |s: &[u8]| std::array::from_fn(|i| u32::from_be_bytes(s[28 - 4 * i..32 - 4 * i].try_into().unwrap()));
    (f(&b[1..33]), f(&b[33..65]))
}

/// pattern -> (mask, value) over 5 words; word w holds address bytes 4w..4w+3, byte 4w lowest
fn gpu_mask(pat: &[(usize, u8)]) -> ([u32; 5], [u32; 5]) {
    let (mut mask, mut val) = ([0u32; 5], [0u32; 5]);
    for &(pos, v) in pat {
        let byte = pos / 2;
        let bit = 8 * (byte % 4) + if pos % 2 == 0 { 4 } else { 0 };
        mask[byte / 4] |= 0xf << bit;
        val[byte / 4] |= (v as u32) << bit;
    }
    (mask, val)
}

fn buffer(ctx: &Context, flags: u64, data: &[u32]) -> Res<Buffer<u32>> {
    let ptr = data.as_ptr() as *mut std::ffi::c_void;
    Ok(unsafe { Buffer::<u32>::create(ctx, flags | CL_MEM_COPY_HOST_PTR, data.len(), ptr)? })
}

/// Search keys k0 + g*2^STRIDE_BITS + i on the GPU. Returns the matching private key.
fn search(k0: Scalar, pat: &[(usize, u8)], n: usize, rounds: u32, max_launches: u64, verbose: bool) -> Res<Option<Scalar>> {
    let device = Device::new(*get_all_devices(CL_DEVICE_TYPE_GPU)?.first().ok_or("no OpenCL GPU found")?);
    if verbose {
        eprintln!("GPU: {}", device.name()?);
    }
    let ctx = Context::from_device(&device)?;
    let queue = CommandQueue::create_default(&ctx, 0)?;
    let program = Program::create_and_build_from_source(&ctx, include_str!("kernel.cl"), &format!("-D M={M}"))?;
    let kernel = Kernel::create(&program, "search")?;

    // start points (k0 + g*stride)*G, built with point additions + one batch inversion
    let stride = (ProjectivePoint::GENERATOR * Scalar::from(1u64 << STRIDE_BITS)).to_affine();
    let mut cur = ProjectivePoint::GENERATOR * k0;
    let pts: Vec<ProjectivePoint> = (0..n)
        .map(|_| {
            let p = cur;
            cur += stride;
            p
        })
        .collect();
    let (mut px, mut py) = (vec![0u32; 8 * n], vec![0u32; 8 * n]);
    for (g, p) in ProjectivePoint::batch_normalize(pts.as_slice()).iter().enumerate() {
        let (x, y) = limbs(p);
        for i in 0..8 {
            px[i * n + g] = x[i];
            py[i * n + g] = y[i];
        }
    }
    // table j*G, j = 1..=M
    let (mut tx, mut ty) = (Vec::with_capacity(8 * M), Vec::with_capacity(8 * M));
    let mut jg = ProjectivePoint::GENERATOR;
    for _ in 0..M {
        let (x, y) = limbs(&jg.to_affine());
        tx.extend(x);
        ty.extend(y);
        jg += AffinePoint::GENERATOR;
    }
    let (mask, val) = gpu_mask(pat);

    let b_px = buffer(&ctx, CL_MEM_READ_WRITE, &px)?;
    let b_py = buffer(&ctx, CL_MEM_READ_WRITE, &py)?;
    let b_tx = buffer(&ctx, CL_MEM_READ_ONLY, &tx)?;
    let b_ty = buffer(&ctx, CL_MEM_READ_ONLY, &ty)?;
    let b_mask = buffer(&ctx, CL_MEM_READ_ONLY, &mask)?;
    let b_val = buffer(&ctx, CL_MEM_READ_ONLY, &val)?;
    let b_found = buffer(&ctx, CL_MEM_READ_WRITE, &[0u32; 3])?;
    let n32 = n as u32;

    let per_launch = n as u64 * rounds as u64 * M as u64;
    let expected = 16f64.powi(pat.len() as i32);
    let t0 = Instant::now();
    for launch in 0..max_launches {
        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&b_px)
                .set_arg(&b_py)
                .set_arg(&b_tx)
                .set_arg(&b_ty)
                .set_arg(&b_mask)
                .set_arg(&b_val)
                .set_arg(&rounds)
                .set_arg(&n32)
                .set_arg(&b_found)
                .set_global_work_size(n)
                .enqueue_nd_range(&queue)?;
        }
        let mut found = [0u32; 3];
        unsafe { queue.enqueue_read_buffer(&b_found, CL_BLOCKING, 0, &mut found, &[])? };
        if found[0] == 1 {
            let offset = launch * rounds as u64 * M as u64 + found[2] as u64;
            let key = k0 + Scalar::from(1u64 << STRIDE_BITS) * Scalar::from(found[1] as u64) + Scalar::from(offset);
            return Ok(Some(key));
        }
        if verbose {
            let done = (launch + 1) as f64 * per_launch as f64;
            let secs = t0.elapsed().as_secs_f64();
            let rate = done / secs;
            let eta = (expected * std::f64::consts::LN_2 - done).max(0.0) / rate;
            eprint!("\r{:.2} G keys | {:.0} M keys/s | {:.0}s elapsed | ~{:.0}s left (50%)   ", done / 1e9, rate / 1e6, secs, eta);
        }
    }
    Ok(None)
}

fn usage() -> ! {
    eprintln!("usage: bsc-vanity [--suffix HEX] [--prefix HEX]\n  e.g. bsc-vanity --suffix 999999");
    process::exit(1)
}

fn prompt(msg: &str) -> String {
    eprint!("{msg}");
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).ok();
    s.trim().to_string()
}

fn main() {
    // double-clicked (no args): ask for the pattern and keep the window open at the end
    let interactive = env::args().len() == 1;
    let (mut prefix, mut suffix) = (String::new(), String::new());
    if interactive {
        suffix = prompt("Suffix (e.g. 66666666): ").to_lowercase();
        prefix = prompt("Prefix (optional, Enter to skip): ").trim_start_matches("0x").to_lowercase();
    }
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--suffix" | "-s" => suffix = val().to_lowercase(),
            "--prefix" | "-p" => prefix = val().trim_start_matches("0x").to_lowercase(),
            _ => usage(),
        }
    }
    let result = match parse_pattern(&prefix, &suffix) {
        Ok(pat) if pat.is_empty() => Err("empty pattern: give a suffix and/or prefix".into()),
        Ok(pat) => run(&prefix, &suffix, &pat),
        Err(e) => Err(e.into()),
    };
    if let Err(e) = &result {
        eprintln!("\nerror: {e}");
    }
    if interactive {
        prompt("\nPress Enter to exit...");
    }
    process::exit(if result.is_ok() { 0 } else { 1 });
}

fn run(prefix: &str, suffix: &str, pat: &[(usize, u8)]) -> Res<()> {
    eprintln!("searching 0x{prefix}...{suffix} | ~{:.0} keys on average", 16f64.powi(pat.len() as i32));

    let key = search(Scalar::random(&mut OsRng), pat, N, ROUNDS, u64::MAX, true)?.expect("search ended without result");
    // re-derive on CPU: never save a key that doesn't own a matching address
    let addr = key_to_address(&key);
    if !matches(&addr, pat) {
        return Err("GPU reported a key whose address does not match - not saved".into());
    }

    let line = format!(
        "Address: {} | Private key: 0x{}\n",
        checksum(&addr),
        key.to_repr().iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
    println!("\n\n{line}");
    // append, never overwrite earlier finds
    let saved = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("found.txt")
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
    match saved {
        Ok(()) => eprintln!("saved to {}", std::path::absolute("found.txt").unwrap_or_default().display()),
        Err(e) => eprintln!("WARNING: could not save found.txt ({e}) - copy the key above now!"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_helpers() {
        // private key 1 -> well-known address
        let a = address(&AffinePoint::GENERATOR);
        assert_eq!(checksum(&a), "0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf");
        assert!(matches(&a, &parse_pattern("7e5f", "bdf").unwrap()));
        assert!(!matches(&a, &parse_pattern("", "999").unwrap()));
        assert!(parse_pattern("", "xyz").is_err());
    }

    #[test]
    fn gpu_finds_planted_key() {
        // plant a key at work-item 5, launch 2, round 1, table index 10; full 40-hex match
        let (n, rounds) = (1024, 2u32);
        let k0 = Scalar::random(&mut OsRng);
        let offset = 2 * rounds as u64 * M as u64 + M as u64 + 10;
        let key = k0 + Scalar::from(1u64 << STRIDE_BITS) * Scalar::from(5u64) + Scalar::from(offset);
        let hex: String = key_to_address(&key).iter().map(|b| format!("{b:02x}")).collect();
        let pat = parse_pattern(&hex, "").unwrap();
        assert_eq!(search(k0, &pat, n, rounds, 4, false).unwrap(), Some(key));
    }
}
