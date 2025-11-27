# Running Helion on RunPod

This guide provides step-by-step instructions for deploying Helion on RunPod with persistent model cache and identity file storage.

## Prerequisites

- RunPod account ([sign up here](https://www.runpod.io/))
- Basic familiarity with Docker and command-line interfaces
- Hugging Face account (for accessing gated models like Llama)

## Step 1: Create a RunPod Pod

1. **Log in to RunPod** and navigate to the **Pods** section
2. **Click "Deploy"** or **"New Pod"**
3. **Select your GPU template:**
   - Choose a GPU with sufficient VRAM (recommended: RTX 3090, A100, or similar)
   - For Llama 3.1 405B: Minimum 24GB VRAM per block
   - For smaller models: 16GB+ VRAM should work

4. **Configure the pod:**
   - **Container Image**: Use a base CUDA image or start with `nvidia/cuda:11.8.0-devel-ubuntu22.04`
   - **Container Disk**: Set to at least 20GB (for dependencies)
   - **Volume**: Create a new volume for persistent storage (recommended: 100GB+ for model cache)

## Step 2: Set Up Persistent Storage Volumes

### Create Volumes

1. **Navigate to "Volumes"** in RunPod dashboard
2. **Create two volumes:**

   **Volume 1: Model Cache**
   - Name: `helion-model-cache`
   - Size: 100GB+ (adjust based on model size)
   - This will store downloaded model weights

   **Volume 2: Identity Storage**
   - Name: `helion-identity`
   - Size: 1GB (small, just for identity files)
   - This will store your DHT identity for stable peer ID

3. **Note the volume paths** - RunPod typically mounts volumes at `/workspace/volume_name`

## Step 3: Connect to Your Pod

1. **Click on your pod** in the RunPod dashboard
2. **Click "Connect"** to open a terminal session
3. You'll be connected via SSH or web terminal

## Step 4: Mount Persistent Volumes

When creating/editing your pod, ensure the volumes are mounted:

- **Model Cache Volume**: Mount at `/workspace/helion-model-cache` → `/cache` (inside container)
- **Identity Volume**: Mount at `/workspace/helion-identity` → `/workspace/helion-identity` (inside container)

Or manually mount them in the pod settings:
- Volume 1: `/workspace/helion-model-cache` → `/cache`
- Volume 2: `/workspace/helion-identity` → `/workspace/helion-identity`

## Step 5: Install Dependencies

Once connected to your pod, run:

```bash
# Update system packages
apt-get update && apt-get install -y build-essential wget git

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p /opt/conda
rm install_miniconda.sh
export PATH="/opt/conda/bin:${PATH}"

# Accept conda terms
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install Python and PyTorch
conda install -y python~=3.10.12 pip
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2
pip install --no-cache-dir bitsandbytes==0.41.3
pip install --no-cache-dir "transformers>=4.55.0,<4.56.0"
```

## Step 6: Clone and Install Helion

```bash
# Clone the repository
cd /workspace
git clone https://github.com/helion-network/helion-core.git
cd helion-core

# Install Helion
pip install --no-cache-dir -e .
```

## Step 7: Set Up Configuration

### Create Configuration Directory

```bash
# Create config directory
mkdir -p /workspace/helion-identity/.config/helion
```

### Create Configuration File

Create a configuration file at `/workspace/helion-core/config.yml`:

```bash
nano /workspace/helion-core/config.yml
```

Add the following configuration (adjust based on your needs):

```yaml
# Model configuration
converted_model_name_or_path: "meta-llama/Meta-Llama-3.1-405B-Instruct"  # or your model
public_name: "runpod-node-1"  # Your node name

# Networking
port: 31337
# Note: RunPod provides a public IP, but you may need to configure port forwarding
# Check your pod's network settings for the public IP

# Swarm
# initial_peers: []  # Leave empty to connect to public swarm
# new_swarm: false    # Set to true for private swarm

# Model/compute
num_blocks: 2  # Adjust based on your GPU VRAM (or leave unset to auto-calculate)
gpu_memory_limit: "6GB"  # Optional: Limit GPU memory usage (e.g., "4GB", "6GB", "8GiB")
                         # If specified, num_blocks will be calculated based on this limit
device: "cuda:0"
torch_dtype: "auto"
quant_type: "int8"  # or "nf4", "none"
# tensor_parallel_devices: ["cuda:0", "cuda:1"]  # Uncomment for multi-GPU

# Caching - IMPORTANT: Use persistent volume
cache_dir: "/cache"  # This maps to your persistent volume
max_disk_space: "50GB"  # Adjust based on your volume size

# Limits and timeouts
max_batch_size: 8192
attn_cache_tokens: 16384
request_timeout: 180
session_timeout: 1800
step_timeout: 300

# Misc
throughput: "auto"
token: true  # Uses HF CLI token; set to your token string if needed
adapters: []
stats_report_interval: 60

# Identity file - IMPORTANT: Use persistent volume
identity_path: "/workspace/helion-identity/.config/helion/identity.bin"
```

### Set Environment Variables

```bash
# Set cache directory environment variable
export PETALS_CACHE=/cache

# Set Hugging Face token (if using gated models)
export HF_TOKEN="your_huggingface_token_here"
# Or login via CLI:
huggingface-cli login
```

## Step 8: Configure RunPod Network Settings

1. **Get your pod's public IP:**
   - In RunPod dashboard, check your pod's network settings
   - Note the public IP address

2. **Configure port forwarding:**
   - RunPod typically provides port forwarding
   - Configure port `31337` to be accessible
   - Note the public port if different from 31337

3. **Update config.yml** with your public IP:
   ```yaml
   public_ip: "YOUR_RUNPOD_PUBLIC_IP"
   port: 31337  # Or the forwarded port
   ```

## Step 9: Run the Helion Server

### Option 1: Using Config File

```bash
cd /workspace/helion-core
python -m helion.cli.run_server --config config.yml
```

### Option 2: Using Command Line Arguments

```bash
cd /workspace/helion-core
python -m helion.cli.run_server \
    meta-llama/Meta-Llama-3.1-405B-Instruct \
    --cache_dir /cache \
    --identity_path /workspace/helion-identity/.config/helion/identity.bin \
    --num_blocks 2 \
    --device cuda:0 \
    --port 31337 \
    --public_ip YOUR_RUNPOD_PUBLIC_IP \
    --public_name runpod-node-1
```

## Step 10: Verify the Setup

1. **Check server logs** - You should see:
   - Model loading progress
   - DHT connection status
   - Server listening on port 31337
   - Block information

2. **Verify persistent storage:**
   ```bash
   # Check model cache
   ls -lh /cache
   
   # Check identity file
   ls -lh /workspace/helion-identity/.config/helion/
   ```

3. **Check swarm status:**
   - Visit https://health.helion.dev
   - Look for your node name in the swarm

## Step 11: Set Up Auto-Start (Optional)

To automatically start the server when the pod restarts, create a startup script:

```bash
# Create startup script
nano /workspace/start_helion.sh
```

Add:

```bash
#!/bin/bash
export PETALS_CACHE=/cache
export PATH="/opt/conda/bin:${PATH}"
cd /workspace/helion-core
python -m helion.cli.run_server --config config.yml
```

Make it executable:

```bash
chmod +x /workspace/start_helion.sh
```

Configure RunPod to run this script on startup (check RunPod's pod settings for startup command configuration).

## Troubleshooting

### Issue: Model cache not persisting

**Solution:**
- Verify volume is mounted correctly: `ls -la /cache`
- Check volume mount in RunPod pod settings
- Ensure `cache_dir` in config points to `/cache`

### Issue: Identity file not persisting

**Solution:**
- Verify identity volume is mounted: `ls -la /workspace/helion-identity`
- Check that `identity_path` in config points to the mounted volume
- Ensure directory exists: `mkdir -p /workspace/helion-identity/.config/helion`

### Issue: Cannot connect to swarm

**Solution:**
- Check firewall settings in RunPod
- Verify port forwarding is configured
- Check that `public_ip` in config matches your pod's public IP
- Review server logs for connection errors

### Issue: Out of memory

**Solution:**
- Reduce `num_blocks` in config
- Enable quantization: `quant_type: "int8"` or `quant_type: "nf4"`
- Reduce `max_batch_size`
- Use a smaller model

### Issue: Hugging Face authentication

**Solution:**
```bash
# Login via CLI
huggingface-cli login

# Or set token in config
token: "your_token_here"
```

## Best Practices

1. **Monitor disk usage:**
   ```bash
   df -h /cache
   ```

2. **Set appropriate `max_disk_space`** in config to prevent filling the volume

3. **Use identity file** for stable peer ID across restarts

4. **Monitor GPU usage:**
   ```bash
   nvidia-smi
   ```

5. **Check server logs regularly** for errors or warnings

6. **Backup your identity file** if you want to maintain the same peer ID across different pods

## Cost Optimization

- Use **Spot instances** for lower costs
- **Stop the pod** when not in use (volumes persist)
- **Monitor usage** in RunPod dashboard
- Use **appropriate GPU size** for your model

## Additional Resources

- [Helion Documentation](https://github.com/helion-network/helion-core/wiki)
- [RunPod Documentation](https://docs.runpod.io/)
- [Helion Discord](https://discord.gg/KdThf2bWVU)

## Example: Complete Setup Script

Save this as `/workspace/setup_helion.sh` and run it once:

```bash
#!/bin/bash
set -e

echo "Setting up Helion on RunPod..."

# Install dependencies
apt-get update && apt-get install -y build-essential wget git

# Install Miniconda
if [ ! -d "/opt/conda" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
    bash install_miniconda.sh -b -p /opt/conda
    rm install_miniconda.sh
fi
export PATH="/opt/conda/bin:${PATH}"

# Install Python and PyTorch
conda install -y python~=3.10.12 pip
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2
pip install --no-cache-dir bitsandbytes==0.41.3
pip install --no-cache-dir "transformers>=4.55.0,<4.56.0"

# Clone and install Helion
cd /workspace
if [ ! -d "helion-core" ]; then
    git clone https://github.com/helion-network/helion-core.git
fi
cd helion-core
pip install --no-cache-dir -e .

# Create directories
mkdir -p /workspace/helion-identity/.config/helion
mkdir -p /cache

# Set environment
export PETALS_CACHE=/cache
export PATH="/opt/conda/bin:${PATH}"

echo "Setup complete! Configure your config.yml and run the server."
```

Make it executable and run:

```bash
chmod +x /workspace/setup_helion.sh
/workspace/setup_helion.sh
```

