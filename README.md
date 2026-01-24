# trenua

```
  |
  __|   __|  _ \  __ \   |   |   _` |
  |    |     __/  |   |  |   |  (   |
 \__| _|   \___| _|  _| \__,_| \__,_|
```

This is simple English-Ukrainian, bidirectional back and forth, translation CLI application (`tr-en-ua`) written in Rust with using
[TranslateGemma](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) open translation model built on `Gemma 3` by Google ([Gemma Terms of Use](https://ai.google.dev/gemma/terms)),
with automatic language recognision feature, and automatic downloading of default or specified model (if not present yet).

## Installation

Download the latest binary for Linux (for now) from [Releases](https://github.com/vitali2y/trenua/releases), or build from source as below.

### Build from source

#### Build Dependencies

Ubuntu/Debian: for Vulkan GPU acceleration:
```bash
sudo apt install glslc libvulkan-dev
```

```bash
git clone --recursive https://github.com/vitali2y/trenua && cd trenua
# default CPU usage
cargo build --release
# or next with GPU usage
cargo build --release --features vulkan
```

## Usage

### Example with text as an argument
```bash
./target/release/trenua "Слава Україні" \
  --hf-repo mradermacher/translategemma-12b-it-i1-GGUF \
  --hf-model translategemma-12b-it.i1-Q4_K_M.gguf --gpu-layers -1
```

### Example with text from stdin (pipe)
```bash
echo "... long text ..." | ./target/release/trenua 
cat text.txt | ./target/release/trenua
```

### CLI usage
```bash
➜  ./target/release/trenua --help
Usage: trenua [<text>] [-m <model>] [--hf-repo <hf-repo>] [--hf-model <hf-model>] [--gpu-layers <gpu-layers>] [--ctx-size <ctx-size>] [-d]

English <-> Ukrainian translation CLI app using TranslateGemma model

Positional Arguments:
  text              text to translate

Options:
  -m, --model       local path to GGUF model file
  --hf-repo         repo @ Hugging Face (e.g.
                    mradermacher/translategemma-12b-it-i1-GGUF)
  --hf-model        model filename @ Hugging Face (e.g.
                    translategemma-12b-it.i1-Q4_K_M.gguf)
  --gpu-layers      number of layers to offload to GPU (-1 = all, default: 0 =
                    CPU)
  --ctx-size        context size in tokens (default: 8192)
  -d, --debug       enable debug logs
  --help, help      display usage information
➜
```

## License                                                                                                                                                                             

MIT license ([LICENSE](https://github.com/vitali2y/simple-opirv2-router/blob/main/LICENSE) or <http://opensource.org/licenses/MIT>)
