#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{anyhow, bail, Context, Result};
use bincode::Options;
use eframe::egui;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Hamming distance between two consecutive recap frames above which a cut is declared.
const CUT_THRESHOLD: u32 = 16;
/// Positions inside every recap scene used as matching references.
const REF_POINTS: [f64; 5] = [0.15, 0.30, 0.50, 0.70, 0.85];
/// Maximum Hamming distance for a movie frame to be a candidate.
const CAND_MAX_DIST: u32 = 18;
/// Maximum number of (non-adjacent) candidates kept per reference frame.
const CAND_COUNT: usize = 30;
/// A reference frame counts as matched when its distance is <= this value.
const GOOD_DIST: u32 = 14;
/// Looser threshold used when extending a neighbouring scene's alignment.
const LOOSE_DIST: u32 = 20;
/// Sample rate of the intermediate audio segments.
const AUDIO_RATE: i64 = 48000;
/// Name of the hash cache stored next to the full movie.
const CACHE_NAME: &str = "movie_hash.cache";

macro_rules! sv {
    ($($x:expr),* $(,)?) => { vec![$( String::from($x) ),*] };
}

// ---------------------------------------------------------------------------
// Messages between worker thread and GUI
// ---------------------------------------------------------------------------

enum Msg {
    Log(String),
    Progress(f32, String),
    Finished(std::result::Result<(), String>),
}

#[derive(Clone)]
struct Reporter {
    tx: Sender<Msg>,
    ctx: egui::Context,
}

impl Reporter {
    fn log(&self, s: impl Into<String>) {
        let _ = self.tx.send(Msg::Log(s.into()));
        self.ctx.request_repaint();
    }
    fn progress(&self, f: f32, stage: &str) {
        let _ = self.tx.send(Msg::Progress(f.clamp(0.0, 1.0), stage.to_string()));
        self.ctx.request_repaint();
    }
}

#[derive(Clone)]
struct Config {
    movie: PathBuf,
    recap: PathBuf,
    output: PathBuf,
    unflip: bool,
    recap_audio: bool,
    gpu: bool,
}

// ---------------------------------------------------------------------------
// External tools (ffmpeg / ffprobe)
// ---------------------------------------------------------------------------

fn find_tool(name: &str) -> PathBuf {
    let exe_name = if cfg!(target_os = "windows") {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(&exe_name);
            if p.exists() {
                return p;
            }
        }
    }
    if !cfg!(target_os = "windows") {
        for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin"] {
            let p = Path::new(dir).join(&exe_name);
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from(name)
}

#[allow(unused_mut)]
fn tool_cmd(name: &str) -> Command {
    let mut c = Command::new(find_tool(name));
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    c
}

fn p(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn run_ffmpeg(args: &[String]) -> Result<()> {
    let out = tool_cmd("ffmpeg")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .context("failed to launch ffmpeg")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let mut tail: Vec<&str> = err.lines().rev().take(6).collect();
        tail.reverse();
        bail!("ffmpeg failed: {}", tail.join(" | "));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Video probing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct VideoInfo {
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    duration: f64,
    has_audio: bool,
}

impl VideoInfo {
    fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den as f64
    }
}

fn parse_rate(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();
    if let Some((a, b)) = s.split_once('/') {
        let a: u32 = a.trim().parse().ok()?;
        let b: u32 = b.trim().parse().ok()?;
        if a > 0 && b > 0 {
            return Some((a, b));
        }
        return None;
    }
    let v: f64 = s.parse().ok()?;
    if v > 0.0 {
        Some(((v * 1000.0).round() as u32, 1000))
    } else {
        None
    }
}

fn probe(path: &Path) -> Result<VideoInfo> {
    let out = tool_cmd("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .context("failed to launch ffprobe (is FFmpeg installed?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe could not read {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut r_rate: Option<(u32, u32)> = None;
    let mut a_rate: Option<(u32, u32)> = None;
    let mut duration = 0f64;
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            match k.trim() {
                "width" => width = v.trim().parse().unwrap_or(0),
                "height" => height = v.trim().parse().unwrap_or(0),
                "r_frame_rate" => r_rate = parse_rate(v),
                "avg_frame_rate" => a_rate = parse_rate(v),
                "duration" => duration = v.trim().parse().unwrap_or(0.0),
                _ => {}
            }
        }
    }
    if width == 0 || height == 0 {
        bail!("{} has no readable video stream", path.display());
    }
    let (fps_num, fps_den) = r_rate
        .or(a_rate)
        .ok_or_else(|| anyhow!("could not determine frame rate of {}", path.display()))?;
    if duration <= 0.0 {
        bail!("could not determine duration of {}", path.display());
    }
    let aout = tool_cmd("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .context("failed to launch ffprobe")?;
    let has_audio = !String::from_utf8_lossy(&aout.stdout).trim().is_empty();
    Ok(VideoInfo {
        width,
        height,
        fps_num,
        fps_den,
        duration,
        has_audio,
    })
}

// ---------------------------------------------------------------------------
// dHash extraction
// ---------------------------------------------------------------------------

fn dhash_from_gray(px: &[u8], mirror: bool) -> u64 {
    let mut h = 0u64;
    let mut bit = 0;
    for y in 0..8 {
        for x in 0..8 {
            let (a, b) = if mirror {
                (px[y * 9 + (8 - x)], px[y * 9 + (7 - x)])
            } else {
                (px[y * 9 + x], px[y * 9 + x + 1])
            };
            if a > b {
                h |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    h
}

/// Decodes every frame (resampled to constant frame rate so frame N == N / fps)
/// and returns the normal dHash list and (optionally) the mirrored dHash list.
fn hash_video(
    path: &Path,
    info: &VideoInfo,
    want_mirror: bool,
    rep: &Reporter,
    p0: f32,
    p1: f32,
    stage: &str,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let vf = format!(
        "fps={}/{},crop=iw:ih*0.7:0:ih*0.15,scale=9:8:flags=area",
        info.fps_num, info.fps_den
    );
    let mut child = tool_cmd("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args([
            "-an",
            "-sn",
            "-dn",
            "-vf",
            vf.as_str(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to launch ffmpeg (is FFmpeg installed?)")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("could not capture ffmpeg output"))?;
    let mut rd = BufReader::with_capacity(1 << 20, stdout);
    let total_est = (info.duration * info.fps()).max(1.0);
    let mut normal: Vec<u64> = Vec::with_capacity(total_est as usize + 16);
    let mut mirrored: Vec<u64> = Vec::new();
    if want_mirror {
        mirrored.reserve(total_est as usize + 16);
    }
    let mut buf = [0u8; 72];
    loop {
        match rd.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                let _ = child.kill();
                return Err(e.into());
            }
        }
        normal.push(dhash_from_gray(&buf, false));
        if want_mirror {
            mirrored.push(dhash_from_gray(&buf, true));
        }
        if normal.len() % 2000 == 0 {
            let f = (normal.len() as f64 / total_est).min(1.0) as f32;
            rep.progress(p0 + (p1 - p0) * f, stage);
        }
    }
    let _ = child.wait();
    if normal.is_empty() {
        bail!("no frames could be decoded from {}", path.display());
    }
    rep.progress(p1, stage);
    Ok((normal, mirrored))
}

// ---------------------------------------------------------------------------
// Movie hash cache
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct HashCache {
    version: u32,
    file_size: u64,
    modified: u64,
    fps_num: u32,
    fps_den: u32,
    hashes: Vec<u64>,
}

fn file_sig(path: &Path) -> (u64, u64) {
    match fs::metadata(path) {
        Ok(m) => {
            let modified = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (m.len(), modified)
        }
        Err(_) => (0, 0),
    }
}

fn cache_opts() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(1u64 << 31)
}

fn load_cache(movie: &Path, cache_path: &Path, info: &VideoInfo) -> Option<Vec<u64>> {
    let data = fs::read(cache_path).ok()?;
    let c: HashCache = cache_opts().deserialize(&data).ok()?;
    let (size, modified) = file_sig(movie);
    if c.version == 1
        && c.file_size == size
        && c.modified == modified
        && c.fps_num == info.fps_num
        && c.fps_den == info.fps_den
        && !c.hashes.is_empty()
    {
        Some(c.hashes)
    } else {
        None
    }
}

fn save_cache(movie: &Path, cache_path: &Path, info: &VideoInfo, hashes: &[u64]) -> Result<()> {
    let (size, modified) = file_sig(movie);
    let c = HashCache {
        version: 1,
        file_size: size,
        modified,
        fps_num: info.fps_num,
        fps_den: info.fps_den,
        hashes: hashes.to_vec(),
    };
    let data = cache_opts().serialize(&c)?;
    fs::write(cache_path, data)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scene matching
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Scene {
    start: usize, // recap frame index (inclusive)
    end: usize,   // recap frame index (exclusive)
}

/// Linear alignment: movie_time = y0 + s * recap_time.
#[derive(Clone, Copy, Debug)]
struct Line {
    y0: f64,
    s: f64,
    mirrored: bool,
}

fn ham(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

struct MatchCtx<'a> {
    movie: &'a [u64],
    rnorm: &'a [u64],
    rmirr: &'a [u64],
    rfps: f64,
    mfps: f64,
}

impl<'a> MatchCtx<'a> {
    fn rh(&self, r: usize, mirrored: bool) -> u64 {
        if mirrored {
            self.rmirr[r]
        } else {
            self.rnorm[r]
        }
    }

    fn eval(&self, refs: &[usize], line: &Line, w: i64) -> Vec<u32> {
        refs.iter()
            .map(|&r| {
                let x = r as f64 / self.rfps;
                let m = ((line.y0 + line.s * x) * self.mfps).round() as i64;
                best_in_window(self.movie, self.rh(r, line.mirrored), m, w).0
            })
            .collect()
    }
}

fn top_candidates(movie: &[u64], h: u64, max_d: u32, k: usize) -> Vec<(u32, usize)> {
    let mut v: Vec<(u32, usize)> = movie
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| {
            let d = ham(h, m);
            if d <= max_d {
                Some((d, i))
            } else {
                None
            }
        })
        .collect();
    v.sort_unstable();
    let mut out: Vec<(u32, usize)> = Vec::with_capacity(k);
    for (d, i) in v {
        if out
            .iter()
            .all(|&(_, j)| (i as i64 - j as i64).abs() > 8)
        {
            out.push((d, i));
            if out.len() >= k {
                break;
            }
        }
    }
    out
}

fn best_in_window(movie: &[u64], h: u64, center: i64, w: i64) -> (u32, i64) {
    let n = movie.len() as i64;
    let mut best = (u32::MAX, center.clamp(0, n - 1));
    for off in 0..=w {
        for sign in [1i64, -1i64] {
            if off == 0 && sign == -1 {
                continue;
            }
            let i = center + off * sign;
            if i < 0 || i >= n {
                continue;
            }
            let d = ham(h, movie[i as usize]);
            if d < best.0 {
                best = (d, i);
            }
        }
    }
    best
}

fn scene_refs(sc: &Scene) -> Vec<usize> {
    let len = sc.end - sc.start;
    let mut v: Vec<usize> = Vec::new();
    for pt in REF_POINTS {
        let mut i = sc.start + (len as f64 * pt) as usize;
        if i >= sc.end {
            i = sc.end - 1;
        }
        if v.last() != Some(&i) {
            v.push(i);
        }
    }
    v
}

fn fit_line(pts: &[(f64, f64)], mirrored: bool) -> Option<Line> {
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|q| q.0).sum::<f64>() / n;
    let my = pts.iter().map(|q| q.1).sum::<f64>() / n;
    let sxx: f64 = pts.iter().map(|q| (q.0 - mx).powi(2)).sum();
    if sxx < 1e-6 {
        return None;
    }
    let sxy: f64 = pts.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum();
    let mut s = sxy / sxx;
    if !(0.25..=4.0).contains(&s) {
        return None;
    }
    if (s - 1.0).abs() < 0.02 {
        s = 1.0;
    }
    Some(Line {
        y0: my - s * mx,
        s,
        mirrored,
    })
}

fn match_scene(cx: &MatchCtx, sc: &Scene) -> Option<Line> {
    if cx.movie.is_empty() || sc.end <= sc.start {
        return None;
    }
    let refs = scene_refs(sc);
    let n = refs.len();
    let first = refs[0];
    let last = refs[n - 1];
    let mut best: Option<(u32, Line)> = None;

    for mirrored in [false, true] {
        let cf = top_candidates(cx.movie, cx.rh(first, mirrored), CAND_MAX_DIST, CAND_COUNT);
        if cf.is_empty() {
            continue;
        }
        if n == 1 {
            for &(d, m) in &cf {
                if best.map_or(true, |(b, _)| d < b) {
                    best = Some((
                        d,
                        Line {
                            y0: m as f64 / cx.mfps - first as f64 / cx.rfps,
                            s: 1.0,
                            mirrored,
                        },
                    ));
                }
            }
            continue;
        }
        let cl = top_candidates(cx.movie, cx.rh(last, mirrored), CAND_MAX_DIST, CAND_COUNT);
        if cl.is_empty() {
            continue;
        }
        let dr = (last - first) as f64 / cx.rfps;
        for &(d1, m1) in &cf {
            for &(d2, m2) in &cl {
                if m2 <= m1 {
                    continue;
                }
                let dm = (m2 - m1) as f64 / cx.mfps;
                let s = dm / dr;
                if !(0.25..=4.0).contains(&s) {
                    continue;
                }
                let mut score = d1 + d2;
                if best.map_or(false, |(b, _)| score >= b) {
                    continue;
                }
                for &r in &refs[1..n - 1] {
                    let frac = (r - first) as f64 / (last - first) as f64;
                    let pos = m1 as f64 + frac * (m2 - m1) as f64;
                    score += best_in_window(cx.movie, cx.rh(r, mirrored), pos.round() as i64, 4).0;
                }
                if best.map_or(true, |(b, _)| score < b) {
                    best = Some((
                        score,
                        Line {
                            y0: m1 as f64 / cx.mfps - s * (first as f64 / cx.rfps),
                            s,
                            mirrored,
                        },
                    ));
                }
            }
        }
    }

    let (_, orig) = best?;
    let mut chosen = orig;
    if n >= 2 {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for &r in &refs {
            let x = r as f64 / cx.rfps;
            let pred = ((orig.y0 + orig.s * x) * cx.mfps).round() as i64;
            let (d, m) = best_in_window(cx.movie, cx.rh(r, orig.mirrored), pred, 6);
            if d <= GOOD_DIST {
                pts.push((x, m as f64 / cx.mfps));
            }
        }
        if let Some(l) = fit_line(&pts, orig.mirrored) {
            let d0: u32 = cx.eval(&refs, &orig, 2).iter().sum();
            let d1: u32 = cx.eval(&refs, &l, 2).iter().sum();
            if d1 <= d0 {
                chosen = l;
            }
        }
    }
    let d = cx.eval(&refs, &chosen, 2);
    let good = d.iter().filter(|&&x| x <= GOOD_DIST).count();
    if good >= n - n / 5 {
        Some(chosen)
    } else {
        None
    }
}

fn verify_line(cx: &MatchCtx, sc: &Scene, line: &Line) -> bool {
    let refs = scene_refs(sc);
    let n = refs.len();
    let d = cx.eval(&refs, line, 3);
    let good = d.iter().filter(|&&x| x <= LOOSE_DIST).count();
    good >= n - n / 5
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Enc {
    Nvenc,
    VideoToolbox,
    X264,
}

fn enc_name(e: Enc) -> &'static str {
    match e {
        Enc::Nvenc => "h264_nvenc (NVIDIA GPU)",
        Enc::VideoToolbox => "h264_videotoolbox (Apple GPU)",
        Enc::X264 => "libx264 -crf 14 -preset fast (CPU)",
    }
}

fn enc_args(enc: Enc, bitrate: u64) -> Vec<String> {
    match enc {
        Enc::X264 => sv![
            "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p", "-bf", "0"
        ],
        Enc::Nvenc => sv![
            "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "16", "-b:v", "0",
            "-pix_fmt", "yuv420p", "-bf", "0"
        ],
        Enc::VideoToolbox => sv![
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            bitrate.to_string(),
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p"
        ],
    }
}

fn detect_encoder(gpu: bool, bitrate: u64, rep: &Reporter) -> Enc {
    if !gpu {
        return Enc::X264;
    }
    let candidate = if cfg!(target_os = "macos") {
        Enc::VideoToolbox
    } else {
        Enc::Nvenc
    };
    let mut a = sv![
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=640x360:r=24:d=0.5"
    ];
    a.extend(enc_args(candidate, bitrate));
    a.extend(sv!["-f", "null", "-"]);
    match run_ffmpeg(&a) {
        Ok(()) => {
            rep.log(format!("GPU encoder detected: {}", enc_name(candidate)));
            candidate
        }
        Err(_) => {
            rep.log("No supported GPU encoder found, falling back to libx264.");
            Enc::X264
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

struct Job {
    idx: usize,
    frames: i64,
    samples: i64,
    line: Option<Line>,
    y_start: f64,
    span: f64,
}

#[derive(Clone)]
struct RenderCfg {
    movie: PathBuf,
    info: VideoInfo,
    unflip: bool,
    bitrate: u64,
    timescale: u32,
}

fn encode_video_segment(job: &Job, rc: &RenderCfg, enc: Enc, out: &Path) -> Result<()> {
    let fps_s = format!("{}/{}", rc.info.fps_num, rc.info.fps_den);
    let mut a = sv!["-v", "error", "-nostdin", "-y"];
    match &job.line {
        Some(l) => {
            a.extend(sv![
                "-ss",
                format!("{:.6}", job.y_start),
                "-t",
                format!("{:.6}", job.span),
                "-i",
                p(&rc.movie)
            ]);
            let mut vf = if (l.s - 1.0).abs() < 1e-9 {
                "setpts=PTS-STARTPTS".to_string()
            } else {
                format!("setpts=(PTS-STARTPTS)/{:.8}", l.s)
            };
            vf.push_str(&format!(",fps={}", fps_s));
            if l.mirrored && !rc.unflip {
                vf.push_str(",hflip");
            }
            vf.push_str(",tpad=stop_mode=clone:stop_duration=4,format=yuv420p");
            a.extend(sv!["-map", "0:v:0", "-vf", vf]);
        }
        None => {
            a.extend(sv![
                "-f",
                "lavfi",
                "-i",
                format!(
                    "color=c=black:s={}x{}:r={}",
                    rc.info.width, rc.info.height, fps_s
                ),
                "-vf",
                "format=yuv420p"
            ]);
        }
    }
    a.extend(sv!["-frames:v", job.frames.to_string(), "-r", fps_s, "-an"]);
    a.extend(enc_args(enc, rc.bitrate));
    a.extend(sv![
        "-video_track_timescale",
        rc.timescale.to_string(),
        p(out)
    ]);
    run_ffmpeg(&a)
}

fn tempo_chain(speed: f64) -> Vec<f64> {
    let mut v = Vec::new();
    let mut x = speed;
    while x > 2.0 {
        v.push(2.0);
        x /= 2.0;
    }
    while x < 0.5 {
        v.push(0.5);
        x *= 2.0;
    }
    if (x - 1.0).abs() > 1e-9 {
        v.push(x);
    }
    v
}

fn encode_audio_segment(job: &Job, rc: &RenderCfg, out: &Path) -> Result<()> {
    let s = job.samples.max(1);
    let mut a = sv!["-v", "error", "-nostdin", "-y"];
    let mut af = String::new();
    let from_movie = job.line.is_some() && rc.info.has_audio;
    if from_movie {
        let l = job.line.as_ref().unwrap();
        a.extend(sv![
            "-ss",
            format!("{:.6}", job.y_start),
            "-t",
            format!("{:.6}", job.span),
            "-i",
            p(&rc.movie),
            "-map",
            "0:a:0",
            "-vn"
        ]);
        for t in tempo_chain(l.s) {
            af.push_str(&format!("atempo={:.8},", t));
        }
    } else {
        a.extend(sv![
            "-f",
            "lavfi",
            "-t",
            format!("{:.3}", s as f64 / AUDIO_RATE as f64 + 3.0),
            "-i",
            format!("anullsrc=r={}:cl=stereo", AUDIO_RATE)
        ]);
    }
    af.push_str(&format!(
        "aresample={},apad=whole_len={},atrim=end_sample={}",
        AUDIO_RATE, s, s
    ));
    a.extend(sv![
        "-af",
        af,
        "-ac",
        "2",
        "-ar",
        AUDIO_RATE.to_string(),
        "-c:a",
        "pcm_s16le",
        p(out)
    ]);
    run_ffmpeg(&a)
}

fn write_concat_list(path: &Path, files: &[PathBuf]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    for file in files {
        let s = file
            .to_string_lossy()
            .replace('\\', "/")
            .replace('\'', "'\\''");
        writeln!(f, "file '{}'", s)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_all(
    jobs: &[Job],
    rc: &RenderCfg,
    cfg: &Config,
    enc: Enc,
    tmp: &Path,
    total_secs: f64,
    rep: &Reporter,
) -> Result<()> {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let threads = if enc == Enc::X264 { (cpus / 2).max(1) } else { 2 };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| anyhow!("could not create thread pool: {e}"))?;
    let done = AtomicUsize::new(0);
    let total = jobs.len().max(1);
    let stage = "4/4 Cutting & merging final video";

    pool.install(|| {
        jobs.par_iter().try_for_each(|job| -> Result<()> {
            let v = tmp.join(format!("v_{:06}.mp4", job.idx));
            encode_video_segment(job, rc, enc, &v)
                .with_context(|| format!("rendering scene segment {}", job.idx))?;
            if !cfg.recap_audio {
                let w = tmp.join(format!("a_{:06}.wav", job.idx));
                encode_audio_segment(job, rc, &w)
                    .with_context(|| format!("rendering audio segment {}", job.idx))?;
            }
            let d = done.fetch_add(1, Ordering::SeqCst) + 1;
            if d % 4 == 0 || d == total {
                rep.progress(0.55 + 0.43 * (d as f32 / total as f32), stage);
            }
            Ok(())
        })
    })?;

    let vfiles: Vec<PathBuf> = jobs
        .iter()
        .map(|j| tmp.join(format!("v_{:06}.mp4", j.idx)))
        .collect();
    let vlist = tmp.join("list.txt");
    write_concat_list(&vlist, &vfiles)?;

    let mut a = sv![
        "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i",
        p(&vlist)
    ];
    if cfg.recap_audio {
        a.extend(sv!["-i", p(&cfg.recap)]);
    } else {
        let afiles: Vec<PathBuf> = jobs
            .iter()
            .map(|j| tmp.join(format!("a_{:06}.wav", j.idx)))
            .collect();
        let alist = tmp.join("audio_list.txt");
        write_concat_list(&alist, &afiles)?;
        a.extend(sv!["-f", "concat", "-safe", "0", "-i", p(&alist)]);
    }
    a.extend(sv![
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-ar",
        "48000",
        "-t",
        format!("{:.6}", total_secs),
        "-movflags",
        "+faststart",
        p(&cfg.output)
    ]);
    rep.log("Merging all segments into the final file...");
    run_ffmpeg(&a).context("final concat/mux step failed")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_time(secs: f64) -> String {
    let total_ms = (secs * 1000.0).round() as i64;
    let h = total_ms / 3_600_000;
    let m = (total_ms / 60_000) % 60;
    let s = (total_ms / 1000) % 60;
    let ms = total_ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn run_pipeline(cfg: &Config, rep: &Reporter) -> Result<()> {
    rep.log("Checking FFmpeg installation...");
    let ver = tool_cmd("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .output()
        .map_err(|_| {
            anyhow!("FFmpeg was not found. Install FFmpeg and add it to PATH, or place ffmpeg/ffprobe next to this program.")
        })?;
    if !ver.status.success() {
        bail!("FFmpeg is installed but failed to run.");
    }
    if !cfg.movie.exists() {
        bail!("Full movie file does not exist.");
    }
    if !cfg.recap.exists() {
        bail!("Recap video file does not exist.");
    }

    let minfo = probe(&cfg.movie)?;
    let rinfo = probe(&cfg.recap)?;
    rep.log(format!(
        "Full movie: {}x{} @ {:.3} fps, {}",
        minfo.width,
        minfo.height,
        minfo.fps(),
        fmt_time(minfo.duration)
    ));
    rep.log(format!(
        "Recap: {}x{} @ {:.3} fps, {}",
        rinfo.width,
        rinfo.height,
        rinfo.fps(),
        fmt_time(rinfo.duration)
    ));

    // ---------------- Stage 1: full movie ----------------
    let stage1 = "1/4 Analyzing full movie";
    rep.progress(0.0, stage1);
    let cache_path = cfg.movie.with_file_name(CACHE_NAME);
    let movie_hashes: Vec<u64> = match load_cache(&cfg.movie, &cache_path, &minfo) {
        Some(h) => {
            rep.log(format!(
                "Loaded cached movie index ({} frames) - skipping analysis.",
                h.len()
            ));
            rep.progress(0.35, stage1);
            h
        }
        None => {
            rep.log("Indexing every frame of the full movie (first run is slow, the result is cached)...");
            let (h, _) = hash_video(&cfg.movie, &minfo, false, rep, 0.0, 0.35, stage1)?;
            rep.log(format!("Indexed {} movie frames.", h.len()));
            match save_cache(&cfg.movie, &cache_path, &minfo, &h) {
                Ok(()) => rep.log(format!("Cache saved: {}", cache_path.display())),
                Err(e) => rep.log(format!("Warning: could not save cache ({e})")),
            }
            h
        }
    };

    // ---------------- Stage 2: recap ----------------
    let stage2 = "2/4 Analyzing recap & detecting cuts";
    rep.progress(0.35, stage2);
    rep.log("Hashing recap frames (normal + mirrored)...");
    let (rnorm, rmirr) = hash_video(&cfg.recap, &rinfo, true, rep, 0.35, 0.45, stage2)?;
    let nr = rnorm.len();
    let rfps = rinfo.fps();
    let mfps = minfo.fps();

    let mut scenes: Vec<Scene> = Vec::new();
    let mut st = 0usize;
    for i in 1..nr {
        if ham(rnorm[i], rnorm[i - 1]) > CUT_THRESHOLD {
            scenes.push(Scene { start: st, end: i });
            st = i;
        }
    }
    scenes.push(Scene { start: st, end: nr });
    rep.log(format!("Detected {} scenes in the recap.", scenes.len()));

    // ---------------- Stage 3: matching ----------------
    let stage3 = "3/4 Matching scenes";
    rep.progress(0.45, stage3);
    let cx = MatchCtx {
        movie: &movie_hashes,
        rnorm: &rnorm,
        rmirr: &rmirr,
        rfps,
        mfps,
    };
    let counter = AtomicUsize::new(0);
    let total_scenes = scenes.len();
    let mut lines: Vec<Option<Line>> = scenes
        .par_iter()
        .map(|sc| {
            let r = match_scene(&cx, sc);
            let d = counter.fetch_add(1, Ordering::SeqCst) + 1;
            if d % 8 == 0 || d == total_scenes {
                rep.progress(0.45 + 0.08 * (d as f32 / total_scenes as f32), stage3);
            }
            r
        })
        .collect();
    let direct = lines.iter().filter(|l| l.is_some()).count();

    // Fill gaps by extending neighbouring alignments (forward, then backward).
    for i in 1..lines.len() {
        if lines[i].is_none() {
            if let Some(l) = lines[i - 1] {
                if verify_line(&cx, &scenes[i], &l) {
                    lines[i] = Some(l);
                }
            }
        }
    }
    for i in (0..lines.len().saturating_sub(1)).rev() {
        if lines[i].is_none() {
            if let Some(l) = lines[i + 1] {
                if verify_line(&cx, &scenes[i], &l) {
                    lines[i] = Some(l);
                }
            }
        }
    }
    let matched = lines.iter().filter(|l| l.is_some()).count();
    let mirrored_count = lines
        .iter()
        .filter(|l| l.map_or(false, |x| x.mirrored))
        .count();
    let sped = lines
        .iter()
        .filter(|l| l.map_or(false, |x| (x.s - 1.0).abs() > 1e-9))
        .count();
    rep.log(format!(
        "Matched {}/{} scenes ({} directly, {} by continuity). Mirrored: {}. Speed-adjusted: {}.",
        matched,
        scenes.len(),
        direct,
        matched - direct,
        mirrored_count,
        sped
    ));
    rep.progress(0.55, stage3);

    // ---------------- Stage 4: render ----------------
    let stage4 = "4/4 Cutting & merging final video";
    let ofps = mfps;
    let fmap = |b: usize| -> i64 { (b as f64 / rfps * ofps).round() as i64 };
    let amap = |f: i64| -> i64 { (f as f64 / ofps * AUDIO_RATE as f64).round() as i64 };
    let total_frames = fmap(nr);
    let total_secs = total_frames as f64 / ofps;

    let mut jobs: Vec<Job> = Vec::new();
    let mut black_frames = 0i64;
    for (i, sc) in scenes.iter().enumerate() {
        let f0 = fmap(sc.start);
        let f1 = fmap(sc.end);
        let n = f1 - f0;
        if n <= 0 {
            continue;
        }
        let samples = amap(f1) - amap(f0);
        let t0 = sc.start as f64 / rfps;
        let idx = jobs.len();
        match lines[i] {
            Some(l) => jobs.push(Job {
                idx,
                frames: n,
                samples,
                line: Some(l),
                y_start: (l.y0 + l.s * t0).max(0.0),
                span: l.s * (n as f64 / ofps) + 1.0,
            }),
            None => {
                black_frames += n;
                jobs.push(Job {
                    idx,
                    frames: n,
                    samples,
                    line: None,
                    y_start: 0.0,
                    span: 0.0,
                });
            }
        }
    }
    if jobs.is_empty() {
        bail!("Nothing to render.");
    }
    rep.log(format!(
        "Output will be exactly {} ({} frames @ {:.3} fps). Black filler: {} frames ({}).",
        fmt_time(total_secs),
        total_frames,
        ofps,
        black_frames,
        fmt_time(black_frames as f64 / ofps)
    ));

    let timescale = {
        let k = ((10000.0 / minfo.fps_num as f64).ceil() as u32).max(1);
        minfo.fps_num * k
    };
    let bitrate = ((minfo.width as f64 * minfo.height as f64 * ofps * 0.18) as u64)
        .clamp(4_000_000, 100_000_000);
    let rc = RenderCfg {
        movie: cfg.movie.clone(),
        info: minfo.clone(),
        unflip: cfg.unflip,
        bitrate,
        timescale,
    };

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!(
        "recap_matcher_{}_{}",
        std::process::id(),
        nanos
    ));
    fs::create_dir_all(&tmp).context("could not create temp folder")?;
    rep.progress(0.55, stage4);

    let enc = detect_encoder(cfg.gpu, bitrate, rep);
    rep.log(format!("Encoder: {}", enc_name(enc)));
    rep.log(format!("Rendering {} segments in parallel...", jobs.len()));

    let mut result = render_all(&jobs, &rc, cfg, enc, &tmp, total_secs, rep);
    if result.is_err() && enc != Enc::X264 {
        rep.log(format!(
            "GPU encoding failed ({}). Restarting with libx264...",
            result.as_ref().err().map(|e| format!("{e:#}")).unwrap_or_default()
        ));
        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::create_dir_all(&tmp);
        rep.progress(0.55, stage4);
        result = render_all(&jobs, &rc, cfg, Enc::X264, &tmp, total_secs, rep);
    }
    let _ = fs::remove_dir_all(&tmp);
    result?;

    rep.log(format!("Done. Saved to: {}", cfg.output.display()));
    Ok(())
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

struct App {
    movie: Option<PathBuf>,
    recap: Option<PathBuf>,
    output: Option<PathBuf>,
    unflip: bool,
    recap_audio: bool,
    gpu: bool,
    running: bool,
    progress: f32,
    stage: String,
    log: Vec<String>,
    rx: Option<Receiver<Msg>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            movie: None,
            recap: None,
            output: None,
            unflip: true,
            recap_audio: true,
            gpu: true,
            running: false,
            progress: 0.0,
            stage: "Idle".to_string(),
            log: vec!["Ready. Select the full movie, the recap video and an output path.".into()],
            rx: None,
        }
    }
}

const VIDEO_EXTS: [&str; 10] = [
    "mp4", "mkv", "avi", "mov", "m4v", "wmv", "webm", "ts", "flv", "mpg",
];

fn path_text(p: &Option<PathBuf>) -> String {
    p.as_ref()
        .map(|x| x.display().to_string())
        .unwrap_or_else(|| "(not selected)".to_string())
}

impl App {
    fn start(&mut self, ctx: &egui::Context) {
        let (movie, recap, output) = match (&self.movie, &self.recap, &self.output) {
            (Some(a), Some(b), Some(c)) => (a.clone(), b.clone(), c.clone()),
            _ => {
                self.log
                    .push("Please select the movie, the recap and the output path first.".into());
                return;
            }
        };
        let cfg = Config {
            movie,
            recap,
            output,
            unflip: self.unflip,
            recap_audio: self.recap_audio,
            gpu: self.gpu,
        };
        let (tx, rx) = channel::<Msg>();
        let rep = Reporter {
            tx: tx.clone(),
            ctx: ctx.clone(),
        };
        self.rx = Some(rx);
        self.running = true;
        self.progress = 0.0;
        self.stage = "Starting...".to_string();
        self.log.clear();
        self.log.push("Processing started.".into());
        std::thread::spawn(move || {
            let res = run_pipeline(&cfg, &rep).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Finished(res));
            rep.ctx.request_repaint();
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut finished = false;
        if let Some(rx) = &self.rx {
            while let Ok(m) = rx.try_recv() {
                match m {
                    Msg::Log(s) => self.log.push(s),
                    Msg::Progress(f, s) => {
                        self.progress = f;
                        self.stage = s;
                    }
                    Msg::Finished(r) => {
                        match r {
                            Ok(()) => {
                                self.progress = 1.0;
                                self.stage = "Finished".to_string();
                                self.log.push("All done.".into());
                            }
                            Err(e) => {
                                self.stage = "Failed".to_string();
                                self.log.push(format!("ERROR: {e}"));
                            }
                        }
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.running = false;
            self.rx = None;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Recap Matcher - rebuild a recap from the full movie");
            ui.add_space(8.0);

            let idle = !self.running;

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(idle, egui::Button::new("Select Full Movie"))
                    .clicked()
                {
                    if let Some(f) = rfd::FileDialog::new()
                        .add_filter("Video", &VIDEO_EXTS)
                        .pick_file()
                    {
                        self.movie = Some(f);
                    }
                }
                ui.label(path_text(&self.movie));
            });
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(idle, egui::Button::new("Select Recap Video"))
                    .clicked()
                {
                    if let Some(f) = rfd::FileDialog::new()
                        .add_filter("Video", &VIDEO_EXTS)
                        .pick_file()
                    {
                        self.recap = Some(f);
                    }
                }
                ui.label(path_text(&self.recap));
            });
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(idle, egui::Button::new("Select Output Path"))
                    .clicked()
                {
                    if let Some(f) = rfd::FileDialog::new()
                        .add_filter("MP4 video", &["mp4"])
                        .set_file_name("final_recap.mp4")
                        .save_file()
                    {
                        let f = if f.extension().is_none() {
                            f.with_extension("mp4")
                        } else {
                            f
                        };
                        self.output = Some(f);
                    }
                }
                ui.label(path_text(&self.output));
            });

            ui.add_space(8.0);
            ui.add_enabled_ui(idle, |ui| {
                ui.checkbox(&mut self.unflip, "Un-flip Mirrored Scenes");
                ui.checkbox(&mut self.recap_audio, "Use Recap Audio");
                ui.checkbox(&mut self.gpu, "Use GPU Hardware Acceleration");
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let can_start = idle
                    && self.movie.is_some()
                    && self.recap.is_some()
                    && self.output.is_some();
                if ui
                    .add_enabled(can_start, egui::Button::new("Start Processing"))
                    .clicked()
                {
                    let c = ctx.clone();
                    self.start(&c);
                }
            });

            ui.add_space(8.0);
            ui.add(
                egui::ProgressBar::new(self.progress)
                    .animate(self.running)
                    .text(format!(
                        "{}  -  {:.0}%",
                        self.stage,
                        self.progress * 100.0
                    )),
            );

            ui.add_space(8.0);
            ui.label("Status Log:");
            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .max_height(ui.available_height())
                    .show(ui, |ui| {
                        for l in &self.log {
                            ui.monospace(l);
                        }
                    });
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 640.0])
            .with_min_inner_size([620.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Recap Matcher",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(App::default()))
        }),
    )
}
