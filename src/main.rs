use anyhow::{Context, Result};
use argh::FromArgs;
use dirs::home_dir;
use glob::glob;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{AddBos, LlamaModel, Special, params::LlamaModelParams},
};
use std::{
    io::{self, Read},
    path::PathBuf,
};
use tracing_subscriber;

#[derive(FromArgs, Debug)]
/// English <-> Ukrainian translation CLI app using TranslateGemma model
struct Args {
    /// local path to GGUF model file
    #[argh(option, short = 'm')]
    model: Option<String>,

    /// repo @ Hugging Face (e.g. mradermacher/translategemma-12b-it-i1-GGUF)
    #[argh(option)]
    hf_repo: Option<String>,

    /// model filename @ Hugging Face (e.g. translategemma-12b-it.i1-Q4_K_M.gguf)
    #[argh(option)]
    hf_model: Option<String>,

    /// number of layers to offload to GPU (-1 = all, default: 0 = CPU)
    #[argh(option, default = "0")]
    gpu_layers: i32,

    /// context size in tokens (default: 16384)
    #[argh(option, default = "16384")]
    ctx_size: u32,

    /// enable debug logs
    #[argh(switch, short = 'd')]
    debug: bool,

    /// text to translate
    #[argh(positional)]
    text: Option<String>,
}

fn main() {
    let args: Args = argh::from_env();

    if args.debug {
        tracing_subscriber::fmt::init();
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(true),
        );
    } else {
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );
    }

    if let Err(e) = run(args) {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    let backend = LlamaBackend::init()?;

    let input_text = match &args.text {
        Some(t) => t.clone(),
        None => {
            let mut buffer = String::new();
            io::stdin()
                .read_to_string(&mut buffer)
                .context("failed to read from stdin")?;
            buffer
        }
    };

    let input_text = input_text.trim().to_string();
    if input_text.is_empty() {
        return Ok(());
    }

    // build prompt
    let prompt = if detect_ukrainian(&input_text) {
        format!("Ukrainian: {}\nEnglish:", input_text)
    } else {
        format!("English: {}\nUkrainian:", input_text)
    };

    // resolve model path
    let model_path = resolve_model_path(args.model, args.hf_repo, args.hf_model, args.debug)?;

    // GPU layers
    let n_gpu_layers = if args.gpu_layers == -1 {
        999
    } else {
        args.gpu_layers
    };
    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers as u32);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .context("failed to load model")?;

    // tokenize
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .context("tokenization failed")?;

    // context
    if args.ctx_size < 2048 {
        return Err(anyhow::anyhow!("context size must be at least 2048"));
    }
    let ctx_params = LlamaContextParams::default()
        .with_n_threads(num_cpus::get() as i32)
        .with_n_ctx(Some(std::num::NonZeroU32::new(args.ctx_size).unwrap()));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("failed to create context")?;

    // decode prompt
    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(token, i as i32, &[0], is_last)?;
    }
    ctx.decode(&mut batch).context("prompt decode failed")?;

    // sampling
    let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
        llama_cpp_2::sampling::LlamaSampler::dist(1234),
        llama_cpp_2::sampling::LlamaSampler::top_p(0.95, 1),
        llama_cpp_2::sampling::LlamaSampler::greedy(),
    ]);
    // let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
    //     llama_cpp_2::sampling::LlamaSampler::dist(1234),
    //     llama_cpp_2::sampling::LlamaSampler::greedy(),
    // ]);

    let mut output = String::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let max_tokens = 8192;
    let mut n_cur = tokens.len() as i32;

    while n_cur < max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut s = String::with_capacity(16);
        let _ = decoder.decode_to_string(&bytes, &mut s, false);

        println!("s: {s}");
        if s.contains("EOF") {
            break; // stop when we have a complete sentence in Ukrainian
        }
        // if s.contains('\n') {
        //     break; // stop at first newline
        // }

        output.push_str(&s);
        n_cur += 1;

        batch.clear();
        batch.add(token, n_cur - 1, &[0], true)?;
        ctx.decode(&mut batch).context("Decode failed")?;
    }
    // while n_cur < max_tokens {
    //     let token = sampler.sample(&ctx, batch.n_tokens() - 1);
    //     sampler.accept(token);

    //     if model.is_eog_token(token) {
    //         break;
    //     }

    //     let bytes = model.token_to_bytes(token, Special::Tokenize)?;
    //     let mut s = String::with_capacity(16);
    //     let _ = decoder.decode_to_string(&bytes, &mut s, false);

    //     // Remove the newline check that was causing early termination
    //     output.push_str(&s);
    //     n_cur += 1;

    //     batch.clear();
    //     batch.add(token, n_cur - 1, &[0], true)?;
    //     ctx.decode(&mut batch).context("Decode failed")?;
    // }

    println!("{}", output.trim());
    Ok(())
}

fn detect_ukrainian(text: &str) -> bool {
    let cyrillic_count: usize = text
        .chars()
        .filter(|c| (0x0400..=0x04FF).contains(&(*c as u32)))
        .count();
    let total = text.chars().filter(|c| !c.is_whitespace()).count();
    total > 0 && cyrillic_count > total / 2
}

fn resolve_model_path(
    model: Option<String>,
    hf_repo: Option<String>,
    hf_model: Option<String>,
    debug: bool,
) -> Result<PathBuf> {
    match (model, hf_repo, hf_model) {
        (Some(path), None, None) => {
            let p = PathBuf::from(path);
            if !p.exists() {
                return Err(anyhow::anyhow!("model file not found: {}", p.display()));
            }
            Ok(p)
        }
        (None, Some(repo), Some(filename)) => {
            // check cache first
            if let Some(cached) = find_cached_hf_model(&repo, &filename)? {
                if debug {
                    eprintln!("using cached model: {}", cached.display());
                }
                return Ok(cached);
            }

            // download silently unless debug
            if debug {
                eprintln!("downloading {}/{}...", repo, filename);
            }
            let api = ApiBuilder::new()
                .with_progress(debug)
                .build()
                .context("failed to init Hugging Face API")?;
            Ok(api.model(repo).get(&filename)?)
        }
        (None, None, None) => {
            // auto mode
            if let Some(p) = try_find_cached_model()? {
                if debug {
                    eprintln!("using auto-detected model: {}", p.display());
                }
                Ok(p)
            } else {
                if debug {
                    eprintln!("downloading default model...");
                }
                let api = ApiBuilder::new()
                    .with_progress(debug)
                    .build()
                    .context("failed to init Hugging Face API")?;
                Ok(api
                    .model("mradermacher/translategemma-12b-it-i1-GGUF".to_string())
                    .get("translategemma-12b-it.i1-Q4_K_M.gguf")?)
            }
        }
        _ => Err(anyhow::anyhow!(
            "specify either `--model <path>` or both `--hf-repo` and `--hf-model`"
        )),
    }
}

// broad search for any TranslateGemma model
fn try_find_cached_model() -> Result<Option<PathBuf>> {
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("Home dir not found"))?;
    let pattern = home.join(
        ".cache/huggingface/hub/models--*--translategemma-*-GGUF/snapshots/*/translategemma-*.gguf",
    );
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid glob pattern"))?;
    let paths: Vec<PathBuf> = glob(pattern_str)?.filter_map(|res| res.ok()).collect();
    Ok(paths.into_iter().next())
}

// precise lookup for specific repo + filename
fn find_cached_hf_model(repo: &str, filename: &str) -> Result<Option<PathBuf>> {
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("Home dir not found"))?;
    let repo_normalized = repo.replace('/', "--");
    let pattern = home.join(format!(
        ".cache/huggingface/hub/models--{}/snapshots/*/{}",
        repo_normalized, filename
    ));
    let pattern_str = pattern
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid glob pattern"))?;
    let paths: Vec<PathBuf> = glob(pattern_str)?.filter_map(|res| res.ok()).collect();
    Ok(paths.into_iter().next())
}
