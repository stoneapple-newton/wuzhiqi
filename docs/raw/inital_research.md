# Executive Summary  
Building a strong Gomoku (五子棋) AI is best approached with an AlphaZero-style pipeline: iteratively self-play to generate training data, train a policy/value neural network, and integrate it with Monte Carlo Tree Search (MCTS) for move selection.  State-of-the-art Gomoku projects (e.g. **Katagomo** and **Rapfi**) use this approach, yielding superhuman play given enough compute【8†L237-L241】【35†L39-L47】.  We recommend starting with a small convolutional policy/value net on the local RTX 4070 (8 GB) for prototyping (e.g. 64–128 channels, few residual blocks) and scaling to larger nets on server GPUs (e.g. 256–512 channels, 10–20 blocks) as resources allow.  Training will be by **self-play reinforcement learning**: generate millions of games by MCTS, store (state, policy, value) examples, then train the network.  Key hyperparameters (learning rate ~1e-3, MCTS simulations 400–800, exploration constant $c_{puct}\approx$1–4, temperature schedule) follow AlphaZero conventions【55†L341-L349】【13†L375-L381】.  Evaluation uses ELO and win-rate vs strong baselines (e.g. Embryo, Katagomo) and tournament benchmarks (Gomocup/Botzone).  We will leverage open-source codebases (e.g. *junxiaosong/AlphaZero_Gomoku*, *Nagi-ovo/AlphaZero-Gomoku*, KataGo fork for Gomoku) and RL frameworks (PyTorch, Ray RLlib).  The pipeline is containerized (Docker), uses mixed precision (FP16) and gradient accumulation on the 4070, and multi-GPU/data-parallel on servers. Experiment tracking (Weights & Biases, TensorBoard) and frequent checkpoints ensure reproducibility.  

【51†embed_image】 *Figure: Monte Carlo Tree Search (MCTS) repeatedly selects, expands, simulates (rolls out), and backs up game states to guide move selection.  AlphaZero integrates MCTS with a neural policy/value network to focus search on promising moves.  (Image: R. Moss, CC BY-SA 4.0【50†L133-L138】.)*  

## 1. Training Pipeline (Step-by-Step)  
We propose an iterative workflow with four main stages (see diagram below). Each **iteration** (epoch) generates new training games and refines the model:  

- **a. Initialize/Load Model:** Define a policy/value CNN (e.g. an 8–10 block ResNet). Optionally load a pretrained or random-weight model. Set up hyperparameters (LR, MCTS sims, etc.)【55†L325-L333】【55†L363-L371】.  
- **b. Self-Play Game Generation:** Run **self-play** using MCTS guided by the current network. For each move in each game, perform *X* MCTS simulations (e.g. 400–800) to build a search tree, then sample a move from the visit-count distribution (with temperature)【13†L375-L381】【20†L250-L258】. Store `(state, π, z)` triples, where π is the MCTS move-probabilities and z is the game outcome (+1 win, −1 loss). Use data augmentation (rotate/flip symmetric moves). Quickly generate thousands of games: on a single 4070, 2 days produced ~2500 games at 8×8【13†L375-L381】; on powerful servers, one can generate >10^5 games/week.  Self-play can be parallelized over CPU cores or GPUs (each worker runs independent MCTS with batched network inference).  
- **c. Neural Network Training:** Train (or fine-tune) the policy/value network on the accumulated self-play data. Use a replay buffer (e.g. last 50k–200k positions【55†L333-L341】) with mini-batches (e.g. 256) and cross-entropy + MSE loss (policy & value heads). Typical settings: Adam or SGD, LR ~10^-3 decaying, a few epochs per iteration【55†L325-L333】. Use mixed precision (FP16) and gradient accumulation to fit batches in 8 GB VRAM. Checkpoint frequently.  
- **d. Evaluation and Selection:** Evaluate the new model by playing (e.g. 40–100) games against the previous best model or a baseline agent (MCTS-only or Embryo). Compute win-rate/ELO. If the new net “beats” the old (e.g. >55% wins), promote it as current model. Record metrics (loss curves, ELO) in logs/W&B/TensorBoard. Then repeat from (b) for the next iteration. Optionally adjust hyperparameters or network size if training stagnates.  

```mermaid
flowchart TB
    A[Initialize or load current network] --> B[MCTS self-play generate games]
    B --> C[Collect (state,π,z) examples into buffer]
    C --> D[Train neural network on buffer]
    D --> E[Evaluate new model vs previous/baseline]
    E --> F{Strong enough?}
    F -->|Yes| G[Save new model as best and continue]
    F -->|No| D
    G --> B
```  

**Compute/Time Estimate:** On the 4070 alone (8 GB), start with a tiny board (e.g. 9×9) to get results in ~1–2 days【13†L375-L381】. A full 15×15 training with a medium network may take weeks on one GPU. With multi-GPU server (e.g. 4×32 GB GPUs), you can parallelize self-play and train large networks (hundreds of millions of positions per day). For example, Katago forks (b28c512 nets) trained over ~10^6 games on clusters to reach high Elo【8†L237-L241】【35†L39-L47】. We budget ~100–500 GPU-hours on servers per week for serious training.  

## 2. Model Architectures (Policy/Value Networks)  
We recommend convolutional residual networks like those in AlphaZero/KataGo, sized to fit VRAM. Key design: input features (binary planes for black/white, maybe liberties), several conv layers + residual blocks, then separate policy and value heads. Below we compare example configurations:

| Model | Residual Blocks | Filters/Width | Params (approx) | VRAM (8GB) | Comments |
|:-----:|:---------------:|:-------------:|:---------------:|:----------:|:---------|
| **Tiny**  | 4 &plusmn; 2 conv  | 64 channels    | ~0.2M | fits easily (∼2GB)  | Good for prototyping 9×9. Quick training, low strength. |
| **Small** | 8 blocks (ResNet) | 64–128 ch     | ~1–2M | ~3–4GB | Baseline for 9×9–11×11. Moderate strength. |
| **Medium**| 10–12 blocks   | 128 ch        | ~4–8M | ~5–6GB | Balanced for 15×15 on 8GB GPU (use FP16, small batches). |
| **Large** | 16–20 blocks   | 192–256 ch    | ~15–30M| requires ≥12GB | For high strength; use server GPUs (A100/RTX 4090). |
| **X-Large** | 20+ blocks    | 256–512 ch    | ≥30M  | ≥24GB   | Cutting-edge like Katago (b28c512) – only on servers. |

All networks output a scalar value ([-1,+1]) and a softmax policy (moves).  For example, our minimal net for 9×9 might be 8 ResNet blocks with 128 filters, roughly 2–4M parameters, needing ~4 GB VRAM (FP32).  The larger “X-Large” nets (e.g. KataGo b28c512 or Rapfi’s ResNet-6x256) have tens of millions of params【34†L67-L70】 and require high-end GPUs or multi-GPU.

【36†embed_image】 *Figure: Example training curves (total loss, policy loss, value loss, learning rate) from an AlphaZero-Gomoku training run on a 3090 Ti. Loss steadily decreases, showing learning. (From Nagi-ovo/AlphaZero-Gomoku.)* 

## 3. Training Algorithms and Data Generation  
**Self-Play RL (AlphaZero):** We use tabula-rasa self-play (no human data needed). Each iteration, play many games where both sides use MCTS+NN.  After each game, store all board states, the MCTS-derived move probabilities (policy target), and final win (value target). Use a replay buffer (ring of last K games)【55†L333-L341】. As training progresses, the network’s policy guides MCTS, creating better self-play data.

**Supervised Bootstrapping (optional):** If human Gomoku game records are available, one could pretrain the network on (state→move) via imitation learning to jump-start training. However, strong Gomoku datasets are scarce. Most effective is pure RL/self-play.

**Reinforcement Learning Variants:** The canonical method is *AlphaZero’s* algorithm (Policy+Value net + MCTS + self-play learning).  Other RL methods like PPO or A3C are less natural for large discrete games, but could be tried on small boards or as experiments.  For Gomoku we focus on MCTS: it handles the combinatorial game tree effectively.  (Rapfi shows that even a depth-first search can exploit a fast NN, but MCTS is standard【35†L39-L47】.)

**MCTS Integration:** Use PUCT (UCB variant) to select moves in the tree: $\mathrm{UCT} = Q + c_{puct} P \sqrt{N}/(1+N_v)$, where $P$ is NN policy prior.  We use a high $c_{puct}$ (e.g. 1–4) to encourage exploration【55†L363-L371】.  Typically each move in self-play runs 400–800 MCTS sims (fewer at start, more as model improves).  Server GPUs can afford thousands.  During inference (play out games or evaluation), use a fixed policy (tau=0) and more simulations for stability.

**Data Augmentation:** Use board symmetries: rotate 90°, 180°, 270° and reflect, to augment training examples (since Gomoku is symmetric). Also shuffle colors for balanced data. 

## 4. Compute Allocation (Local vs Server)  
We outline tasks and recommended hardware:

| Task                      | Local RTX 4070 (8GB)           | Server GPU(s) (≥24GB, multi-GPU)        |
|:--------------------------|:-------------------------------|:----------------------------------------|
| **Self-play generation**  | Small-scale: run ~10–100 games/day (caps MCTS sims ~100). Good for debugging and initial data.  | Large-scale: parallel workers (1 per GPU/CPU) generating 10^3–10^4 games/day. High MCTS sims per move (800–2000) and batched inference. |
| **Model training**        | Small nets (≤64ch) with small batches (≈64–128) – can train basic nets. Use mixed precision.  | Full training: large nets (128–512ch), big batches (256–1024) with gradient accumulation or multi-GPU (DataParallel/Distributed). Use FP16 AMP to fit memory. |
| **Evaluation / Arena**    | Quick tests: play head-to-head matches or measure win-rate of 1–2 k games (CPU+4070).  | Full tournaments: run long evaluation (several thousand games) in parallel. Use dedicated evaluation GPUs. |
| **Experiment tracking**   | TensorBoard/W&B on local, small experiments. | Centralized logging (W&B) integrated across nodes. |
| **Infrastructure**        | Docker container with PyTorch, RLlib, etc. Local CPU/CPU clusters for lightweight tasks. | Containerized training on Linux cluster; use Slurm/Ray for distributed jobs. Multi-node orchestration via Docker/Kubernetes if available. |

We **batch** as much as possible: e.g. during self-play, run MCTS in parallel threads, with each neural net inference called with a batch of states to amortize GPU use.  Use AMP (automatic mixed precision) for FP16 to halve memory and speed up throughput.  Save frequent **checkpoints** (every few hours) to survive interruptions.

Leverage multi-GPU frameworks: e.g. PyTorch DistributedDataParallel or Ray RLlib’s multi-worker support, to train across GPUs.  On the 4070 (single GPU), use `torch.nn.DataParallel` or accumulate gradients to simulate larger batch.  

## 5. Key Resources (Code, Libraries, Papers)  
- **AlphaZero/Gomoku implementations:**  
  – *Junxiaosong/AlphaZero_Gomoku* (PyTorch/Theano) – a classic Alphazero tutorial repo【13†L298-L302】.  
  – *Nagi-ovo/AlphaZero-Gomoku* (PyTorch, self-play with replay, uses wandb, 15×15 and 9×9 support)【32†L274-L282】【32†L323-L332】.  
  – *Kuangliu/gomoku-ai* or *Darter101/Gomoku* (some policy-gradient examples).  
  – *KataGomo* (fork of KataGo for Gomoku) – highly optimized C++/CUDA engine (GitHub: hzyhhzy/KataGomo)【8†L237-L241】.  
- **Frameworks/Libraries:**  
  – [PyTorch](https://pytorch.org) or [TensorFlow](https://tensorflow.org) – for NN training. PyTorch is widely used for RL.  
  – [OpenSpiel](https://github.com/deepmind/open_spiel) – DeepMind’s RL library (includes Gomoku implementation).  
  – [Ray RLlib](https://docs.ray.io/en/latest/rllib.html) – scalable RL (has AlphaZero-like single-agent support).  
  – [Leela Zero](https://github.com/leela-zero/leela-zero) – Go RL engine; not directly Gomoku but its efficient implementation (MCTS+NN) can inspire optimizations.  
  – [KataGo](https://github.com/lightvector/KataGo) – advanced Go engine; use it as a template for large CNN + MCTS code (its fork KataGomo is for Gomoku).  
- **Open-Source Models:**  
  – *Katago Training* nets (e.g. `kata1-b28c512nbt`) – networks for Go, but same architecture; can adapt code.  
  – *Leela Zero networks* – strong Go nets (architecture inspiration).  
- **Key Papers & Tutorials:**  
  – Silver *et al.*, *AlphaGo Zero* and *AlphaZero*【15†L111-L114】 – foundational algorithms.  
  – Liang *et al.*, *AlphaZero Gomoku* (2023) – applied AlphaZero to Gomoku, showing 100% first-player wins in experiments【15†L111-L114】【20†L250-L254】.  
  – Rapfi (2025, arXiv) – efficient Gomoku network (MixNet) outperforming traditional CNNs under resource limits【35†L39-L47】.  
  – *Katago/Gomoku (Katagomo)* documentation or blog posts for hyperparams.  
  – Medium/Tutorials: e.g. “AlphaZero from Scratch” (sent by Nagi-ovo), Kaggle notebooks.  
  – Gomoku programming contests (Gomocup) discussions for hardware setups.  

## 6. Hyperparameters and Training Details  
We suggest starting with these default settings (to be tuned):  

- **Learning:** Adam optimizer, initial LR ≈1e-3 decayed linearly or with cosine schedule over epochs. Nagi’s config uses 1e-4–1e-2 with 1-cycle schedule【55†L353-L361】. Use weight decay ≈1e-4, gradient clipping ~1.  
- **Batch size:** 64–256 per GPU (accumulate if limited). We used 256 in config【55†L325-L333】 (with accumulation on 4070).  
- **MCTS sims per move:** 400–800 at start; increase to 1000+ as model improves. Use more sims on server.  
- **cpuct:** 1–4.  Nagi’s code uses 4.0【55†L363-L371】. Higher encourages exploration.  
- **Temperature (τ):** Use τ=1 (randomized moves) for early moves (first 15–30 moves) then τ=0 (greedy) thereafter【55†L341-L344】. This prevents deterministic play.  
- **Replay buffer:** 50k–200k most recent states (ring buffer). Train new net on mini-batches sampled from buffer. Use prioritization by game outcome if desired.  
- **Iteration frequency:** Run e.g. 100–200 self-play games per iteration (or 1k on server), then train network for ~1–5 epochs, then evaluate. The Nagi code does 100 games ×10 epochs per iter【55†L325-L333】.  
- **Loss:** policy cross-entropy + value mean-squared error (plus L2 reg).  
- **Exploration:** Occasionally add Dirichlet noise to root P-distribution (as in original AlphaZero) to ensure diversity in early moves.  
- **Curriculum:** Optionally start on smaller board or less sims to find a baseline model (as in [13]) then scale up (transfer learning) to full 15×15. For example, first train 6×6 or 9×9, then extend to 15×15 using those weights.  

Tuning strategy: monitor ELO and training loss. If plateau, try increasing network capacity, MCTS sims, or data (self-play games). Use learning rate schedules or early-stopping if overfitting.  

## 7. Evaluation Protocol & Benchmarks  
Evaluate model strength rigorously:  

- **Self-play Elo:** Maintain an Elo ladder of saved models (e.g. using OpenSpiel or any ELO code). Each new model plays ~50–100 games against the previous champion; update Elo (K-factor ~10). Ensure repeatable seeds/conditions.  
- **Win-rate vs Baselines:** Play a large number of games (≥2000) against fixed opponents: random, heuristic (e.g. greedy longest-chain), and known strong engines (Embryo, Katagomo, Deep (Yixin) Gomoku). Compute win/draw/loss rates. A strong model should beat Embryo and approach or exceed Katagomo.  
- **Gomocup/Gomoku tournaments:** Submit trained models to open competitions (Gomocup, Botzone). Rapfi’s ranking on Botzone and Gomocup was used as proof of strength【35†L43-L52】.  
- **Specific test positions:** Create curated “test suites” (like life-and-death problems or trap scenarios) to check if the agent finds the correct winning move.  
- **Speed vs strength trade-off:** For deployment, measure inference time per move (with given MCTS sims) to ensure real-time play feasibility (<<1s for 400 sims on 4070). Possibly distill network or reduce sims for final inference.  

## 8. Starter Plan and Scaling  

**Local RTX 4070 (Minimal Reproducible):**  
- **Environment:** Install Docker with PyTorch and required libs (numpy, wandb, etc). Clone a simple AlphaZero Gomoku repo (e.g. Nagi-ovo’s)【32†L274-L282】.  
- **Small board:** Begin with 9×9 Gomoku. Set network to 8 blocks × 64 filters (≈1M params). Use ~200 MCTS sims, cpuct=1. Generate ~100 games, train for 5–10 epochs, evaluate. Expect a basic player in <1 week.  
- **Observe Metrics:** Track loss curves (like [36]) and win-rates. Tweak hyperparams (LR, batch) as needed. Save checkpoints frequently.  
- **Goal:** Achieve >50% win vs random/greedy and decent play. Use this setup to verify pipeline correctness.  

**Server Multi-GPU (Scaled Plan):**  
- **Expand to 15×15:** Increase network to 12–16 blocks × 128–256 filters (5–15M params). Use 4 GPUs in parallel.  
- **Distributed Self-Play:** Run e.g. 10 parallel workers (each on CPU/GPU) generating games. Collect into a shared buffer (possibly via Ray or file I/O).  
- **Larger Training:** Combine GPUs with DDP to train large nets (batch 512–1024, gradient accumulation). Mixed precision to fit memory.  
- **More Games:** Generate >10^4 games per iteration, buffer size ~100k.  
- **Long runs:** Plan weeks of training. Use Slurm/cluster or Kubernetes.  
- **Evaluation:** Schedule automated tournaments vs top Gomoku engines daily to monitor Elo.  
- **Infrastructure:** Use W&B for logging (as in Nagi’s code), TensorBoard for TensorFlow runs, and Docker images for consistency.  

**Summary:** This pipeline – self-play RL, progressive network scaling, careful hyperparam tuning and evaluation – has proven effective for Gomoku AI【13†L375-L381】【35†L39-L47】. With a methodical approach and the above resources, one can train a Gomoku agent from scratch to strong performance within a few weeks of compute.  

**Key References:** Silver *et al.* (2017) [AlphaZero algorithm]; Liang *et al.* (2023) [AlphaZero Gomoku]【15†L111-L114】【20†L250-L254】; Zhang *et al.* (2025) [Rapfi Gomoku]【35†L39-L47】; open-source code (e.g. *Nagi-ovo/AlphaZero-Gomoku*【32†L274-L282】). These and the cited resources guide the detailed choices above. 

