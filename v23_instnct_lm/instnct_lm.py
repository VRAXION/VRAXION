"""
VRAXION v23 — INSTNCT CPU Language Learning Experiment
======================================================
Byte-level language model with:
- Ring buffer recurrent memory
- Self-wiring sparse connections (edge list + scatter/gather)
- Inverse arousal gate (confident → more wiring, uncertain → less)
- Continuous learning (no train/eval split)
- CPU-optimized PyTorch

Three sizes: Tiny (~0.5M), Small (~2M), Medium (~8M)

Author: Daniel (researcher) + Claude (advisor)
Date: 2026-03-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import os

torch.set_num_threads(min(4, os.cpu_count() or 4))

# ── Inline English corpus ──────────────────────────────────────

CORPUS_SENTENCES = [
    "The cat sat on the mat.",
    "I like to eat apples and oranges.",
    "The sun rises in the east and sets in the west.",
    "Birds can fly high in the sky.",
    "She walked to the store to buy some bread.",
    "He reads a book every evening before bed.",
    "The flowers bloom in the spring.",
    "We went to the park on Sunday afternoon.",
    "The dog chased the ball across the yard.",
    "My mother makes the best chocolate cake.",
    "The rain fell softly on the window.",
    "They played football in the garden.",
    "The teacher explained the lesson clearly.",
    "I drank a glass of cold water.",
    "The children laughed and played together.",
    "She wore a beautiful blue dress.",
    "The car stopped at the red light.",
    "He wrote a letter to his friend.",
    "The stars shine brightly at night.",
    "We had dinner at a nice restaurant.",
    "The baby slept peacefully in the crib.",
    "I finished my homework before dinner.",
    "The birds sang sweetly in the trees.",
    "She painted a picture of the mountains.",
    "The wind blew the leaves off the trees.",
    "He fixed the broken chair in the garage.",
    "The snow covered the ground like a white blanket.",
    "I went swimming in the lake last summer.",
    "The clock on the wall shows the time.",
    "She baked cookies for her neighbors.",
    "The train arrived at the station on time.",
    "He planted tomatoes in the garden.",
    "The moon was full and bright tonight.",
    "We watched a movie at the cinema.",
    "The fish swam in the clear blue water.",
    "I took my umbrella because it might rain.",
    "The old man sat on the bench and fed the pigeons.",
    "She learned to play the piano when she was young.",
    "The river flows through the valley.",
    "He cleaned his room before his friends arrived.",
    "The leaves change color in autumn.",
    "I bought a new pair of shoes yesterday.",
    "The cat climbed up the tall tree.",
    "She made a cup of tea and sat down to read.",
    "The bus comes every fifteen minutes.",
    "He scored the winning goal in the match.",
    "The mountains are covered with snow in winter.",
    "I helped my father wash the car.",
    "The library has thousands of books.",
    "She smiled and waved goodbye.",
    "What is your name? My name is Assistant.",
    "How are you today? I am doing well, thank you.",
    "Where do you live? I live in a small town near the river.",
    "What time is it? It is half past three in the afternoon.",
    "Do you like music? Yes, I enjoy listening to many kinds of music.",
    "What is your favorite color? My favorite color is blue.",
    "How old are you? I am twenty five years old.",
    "What do you do for work? I work as a software engineer.",
    "Can you help me? Of course, I would be happy to help you.",
    "What did you eat for breakfast? I had toast and eggs.",
    "Where is the nearest hospital? It is about two miles down the road.",
    "Do you have any brothers or sisters? I have one brother and two sisters.",
    "What is the weather like today? It is sunny and warm.",
    "When does the store open? The store opens at nine in the morning.",
    "Why are you late? I missed the bus and had to walk.",
    "What are you reading? I am reading a mystery novel.",
    "How do you spell that word? It is spelled with a double letter.",
    "What is the capital of France? The capital of France is Paris.",
    "Do you speak any other languages? I speak English and Spanish.",
    "What would you like to drink? I would like a glass of water please.",
    "How far is it to the airport? It takes about thirty minutes by car.",
    "What is your hobby? I enjoy painting and hiking.",
    "Can I borrow your pen? Sure, here you go.",
    "What happened yesterday? We had a surprise birthday party.",
    "Where did you go on vacation? We visited the mountains last summer.",
    "Please open the door for me.",
    "Turn off the light when you leave the room.",
    "Remember to lock the door before you go to sleep.",
    "Please pass me the salt and pepper.",
    "Do not forget to water the plants.",
    "Make sure you finish your homework before playing.",
    "Please be quiet in the library.",
    "Take your shoes off before entering the house.",
    "Wash your hands before eating.",
    "Please send me the report by Friday.",
    "Close the window because it is getting cold.",
    "Be careful when crossing the street.",
    "Please write your name at the top of the page.",
    "Set the alarm for seven in the morning.",
    "Remember to bring your lunch to school.",
    "Turn left at the next intersection.",
    "Please wait here while I get the car.",
    "Put the dishes in the sink after dinner.",
    "Keep your room clean and organized.",
    "Always wear your seatbelt in the car.",
    "There are seven days in a week.",
    "Water boils at one hundred degrees Celsius.",
    "The Earth goes around the Sun once every year.",
    "There are twelve months in a year.",
    "The human body has two hundred and six bones.",
    "Light travels at about three hundred thousand kilometers per second.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "There are twenty six letters in the English alphabet.",
    "A triangle has three sides and three angles.",
    "The tallest mountain in the world is Mount Everest.",
    "A year has three hundred and sixty five days.",
    "Mars is called the Red Planet.",
    "An octopus has eight arms.",
    "Diamonds are the hardest natural material.",
    "The brain uses about twenty percent of the body energy.",
    "Gold is a chemical element with the symbol Au.",
    "Good morning! How did you sleep? I slept very well, thank you.",
    "Hello! Nice to meet you. Nice to meet you too.",
    "Excuse me, could you tell me the way to the station? Go straight and turn right.",
    "Thank you very much for your help. You are welcome.",
    "I am sorry for being late. No problem, we just started.",
    "See you tomorrow! Have a great evening. You too, goodbye!",
    "Would you like some coffee? Yes please, with milk and sugar.",
    "How was your weekend? It was wonderful, we went to the beach.",
    "Happy birthday! Thank you, it has been a great day.",
    "Congratulations on your new job! Thank you, I am very excited.",
    "I hope you feel better soon. Thank you, I appreciate that.",
    "What a beautiful day! Yes, the weather is perfect for a walk.",
    "I have not seen you in a long time! I know, we should catch up.",
    "Welcome to our home! Thank you for having us.",
    "Let me introduce you to my friend. Hello, pleased to meet you.",
    "I need to go now. Okay, it was nice talking to you.",
    "Can I take your order? I will have the chicken salad please.",
    "The meeting is at three o clock. I will be there on time.",
    "Did you watch the game last night? Yes, it was very exciting.",
    "I am going to the gym. Would you like to join me?",
    "This is my favorite restaurant. The food here is amazing.",
    "I forgot my wallet at home. Do not worry, I can pay for you.",
    "The flight is delayed by two hours. Let us get some coffee while we wait.",
    "I just finished reading a great book. What was it about?",
    "We are having a barbecue on Saturday. That sounds like fun!",
    "Once upon a time, there was a little girl who lived in a small village near a dark forest.",
    "The scientist worked late into the night, carefully recording each measurement.",
    "On a cold winter morning, the farmer woke up early to feed his animals.",
    "The ship sailed across the ocean for many days.",
    "In the middle of the city, there was a beautiful garden where people came to relax.",
    "The student studied hard for the exam. She spent hours reading her notes.",
    "The chef prepared a special meal for the guests using fresh ingredients.",
    "The musician played her guitar softly as the audience listened in silence.",
    "Every Saturday, the family would gather around the table for a big breakfast.",
    "The detective examined the clues carefully. Something about this case did not add up.",
    "The artist spent months working on his masterpiece. Everyone was amazed.",
    "The children built a sandcastle on the beach with tall towers and a moat.",
    "Spring arrived and the garden came alive with color.",
    "The traveler walked along the dusty road, carrying only a small backpack.",
    "The old clock tower in the center of town had been standing for over a hundred years.",
    "She opened the window and let the fresh morning air fill the room.",
    "The firefighters rushed to the scene and quickly put out the flames.",
    "He looked up at the night sky and counted the stars.",
    "The bakery on the corner made the best bread in town.",
    "After a long day of work, she sat down with a good book and hot chocolate.",
    "A computer processes information using billions of tiny switches called transistors.",
    "The internet connects millions of computers around the world.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Gravity is the force that keeps us on the ground.",
    "Sound travels through the air as waves that vibrate at different frequencies.",
    "The human eye can distinguish about ten million different colors.",
    "Electricity flows through wires like water flows through pipes.",
    "A rainbow appears when sunlight is split into its component colors by water droplets.",
    "The seasons change because the Earth is tilted on its axis.",
    "I think we should go for a walk before it gets dark.",
    "She always brings her lunch to work in a brown paper bag.",
    "The movie was so funny that everyone in the theater was laughing.",
    "He decided to learn a new language and chose to study Japanese.",
    "They built a treehouse in the old oak tree in the backyard.",
    "I cannot find my keys anywhere. Have you seen them?",
    "The concert was sold out weeks before the show.",
    "She runs five kilometers every morning before breakfast.",
    "The restaurant on the corner serves the best pizza in town.",
    "He was surprised to find a letter in the mailbox from an old friend.",
    "The children were excited about the field trip to the museum.",
    "I need to buy groceries on my way home from work.",
    "The sunset painted the sky in shades of orange and pink.",
    "She taught herself to code by watching online tutorials.",
    "He always carries a notebook to write down his ideas.",
    "The forest was quiet except for the sound of birds singing.",
    "I woke up early to watch the sunrise from the hilltop.",
    "The team worked together to solve the difficult problem.",
    "The coffee shop on Main Street has the best espresso.",
    "He practiced the violin for two hours every day.",
    "The lake was so clear you could see the fish swimming below.",
    "I am learning to cook new recipes from around the world.",
    "The thunderstorm lasted all night and the morning was fresh and cool.",
    "The market was full of fresh fruits and vegetables.",
    "He walked to school every day regardless of the weather.",
    "The evening was peaceful and the air smelled of flowers.",
]


def build_corpus_bytes():
    """Join all sentences with newlines and convert to byte tensor."""
    text = "\n".join(CORPUS_SENTENCES) + "\n"
    data = list(text.encode('utf-8'))
    return torch.tensor(data, dtype=torch.long), text


# ── Ring Buffer ────────────────────────────────────────────────

class RingBuffer(nn.Module):
    """Fixed-size circular buffer with attention-based reading."""

    def __init__(self, buffer_size, d_model):
        super().__init__()
        self.buffer_size = buffer_size
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def init_buffer(self):
        return torch.zeros(self.buffer_size, self.d_model), 0

    def write(self, buf, ptr, hidden):
        buf = buf.clone()
        buf[ptr % self.buffer_size] = hidden.detach()
        return buf, (ptr + 1) % self.buffer_size

    def read(self, buf, query):
        q = self.query_proj(query).unsqueeze(0)
        k = self.key_proj(buf)
        v = self.value_proj(buf)
        attn = (q @ k.t()) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        return out.squeeze(0)


# ── Self-Wiring Sparse Layer ──────────────────────────────────

class SelfWiringLayer(nn.Module):
    """Sparse layer with edge list + scatter/gather ops."""

    def __init__(self, size, max_edges, init_density=0.1):
        super().__init__()
        self.size = size
        self.max_edges = max_edges

        n_init = min(int(size * size * init_density), max_edges)

        # Pre-allocate edge storage
        src = torch.randint(0, size, (max_edges,))
        dst = torch.randint(0, size, (max_edges,))
        weights = torch.zeros(max_edges)
        weights[:n_init] = torch.randn(n_init) * (2.0 / size) ** 0.5

        self.register_buffer('src', src)
        self.register_buffer('dst', dst)
        self.edge_weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.zeros(size))
        self.n_edges = n_init

        self.register_buffer('last_input', torch.zeros(size))
        self.register_buffer('last_output', torch.zeros(size))
        self.edges_added = 0
        self.edges_removed = 0

    def forward(self, x):
        self.last_input = x.detach()
        n = self.n_edges
        src_vals = x[self.src[:n]]
        weighted = src_vals * self.edge_weights[:n]
        out = torch.zeros(self.size, device=x.device)
        out.scatter_add_(0, self.dst[:n], weighted)
        out = out + self.bias
        self.last_output = out.detach()
        return out

    @torch.no_grad()
    def self_wire(self, n_add=2, n_remove=1):
        if self.n_edges == 0:
            return

        # Remove weakest
        if n_remove > 0 and self.n_edges > self.size:
            n = self.n_edges
            abs_w = self.edge_weights[:n].abs()
            n_rem = min(n_remove, n // 4)
            _, weak_idx = abs_w.topk(n_rem, largest=False)

            keep_mask = torch.ones(n, dtype=torch.bool)
            keep_mask[weak_idx] = False
            n_keep = keep_mask.sum().item()

            self.src[:n_keep] = self.src[:n][keep_mask]
            self.dst[:n_keep] = self.dst[:n][keep_mask]
            self.edge_weights.data[:n_keep] = self.edge_weights.data[:n][keep_mask]
            self.n_edges = n_keep
            self.edges_removed += len(weak_idx)

        # Add new edges
        if n_add > 0 and self.n_edges + n_add <= self.max_edges:
            inp = self.last_input
            out = self.last_output
            active_in = inp.abs().topk(min(8, self.size)).indices
            active_out = out.abs().topk(min(8, self.size)).indices

            added = 0
            for _ in range(n_add * 4):
                if added >= n_add:
                    break
                s = active_in[torch.randint(len(active_in), (1,))].item()
                d = active_out[torch.randint(len(active_out), (1,))].item()

                existing = (self.src[:self.n_edges] == s) & (self.dst[:self.n_edges] == d)
                if not existing.any():
                    idx = self.n_edges
                    self.src[idx] = s
                    self.dst[idx] = d
                    self.edge_weights.data[idx] = torch.randn(1).item() * 0.1
                    self.n_edges += 1
                    added += 1
                    self.edges_added += 1


# ── Processing Block ──────────────────────────────────────────

class ProcessingBlock(nn.Module):
    def __init__(self, d_model, buffer_size, max_edges):
        super().__init__()
        self.sparse = SelfWiringLayer(d_model, max_edges=max_edges)
        self.ring = RingBuffer(buffer_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.act = nn.GELU()

    def forward(self, x, buf, ptr):
        sparse_out = self.act(self.sparse(x))
        ring_out = self.ring.read(buf, x)
        merged = torch.cat([sparse_out, ring_out], dim=-1)
        g = torch.sigmoid(self.gate(merged))
        gated = g * sparse_out + (1 - g) * ring_out
        out = self.norm(x + gated)
        buf, ptr = self.ring.write(buf, ptr, out)
        return out, buf, ptr


# ── INSTNCT Language Model ────────────────────────────────────

class INSTNCTLM(nn.Module):
    def __init__(self, d_model=128, n_blocks=4, buffer_size=256, max_edges=None):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.buffer_size = buffer_size
        if max_edges is None:
            max_edges = d_model * d_model // 2

        self.embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            ProcessingBlock(d_model, buffer_size, max_edges)
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 256)

        # Inverse arousal
        self.arousal_confidence = 0.0
        self.arousal_ema = 0.99
        self.total_edges_added = 0
        self.total_edges_removed = 0

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_active_edges(self):
        return sum(b.sparse.n_edges for b in self.blocks)

    def init_state(self):
        bufs, ptrs = [], []
        for block in self.blocks:
            buf, ptr = block.ring.init_buffer()
            bufs.append(buf)
            ptrs.append(ptr)
        return bufs, ptrs

    def forward_step(self, byte_idx, pos, bufs, ptrs):
        x = self.embed(byte_idx) + self.pos_embed(pos % 512)
        new_bufs, new_ptrs = [], []
        for i, block in enumerate(self.blocks):
            x, buf, ptr = block(x, bufs[i], ptrs[i])
            new_bufs.append(buf)
            new_ptrs.append(ptr)
        x = self.out_norm(x)
        logits = self.out_proj(x)
        return logits, new_bufs, new_ptrs

    def update_arousal(self, loss_val):
        # Use normalized confidence: loss of ~2.5 → 0.0, loss of ~1.0 → 1.0
        # This maps the realistic byte-prediction loss range to [0, 1]
        confidence = max(0.0, min(1.0, (3.0 - loss_val) / 2.0))
        self.arousal_confidence = (
            self.arousal_ema * self.arousal_confidence +
            (1 - self.arousal_ema) * confidence
        )

    def get_wiring_intensity(self):
        """Inverse arousal: confident → more wiring, uncertain → less."""
        if self.arousal_confidence > 0.5:
            return 5, 3  # high confidence: active building
        elif self.arousal_confidence > 0.3:
            return 3, 1  # moderate: some wiring
        elif self.arousal_confidence > 0.1:
            return 1, 0  # low: minimal wiring, no removal
        else:
            return 0, 0  # very uncertain: no wiring

    @torch.no_grad()
    def do_self_wiring(self):
        n_add, n_remove = self.get_wiring_intensity()
        if n_add == 0 and n_remove == 0:
            return
        for block in self.blocks:
            old_a, old_r = block.sparse.edges_added, block.sparse.edges_removed
            block.sparse.self_wire(n_add=n_add, n_remove=n_remove)
            self.total_edges_added += block.sparse.edges_added - old_a
            self.total_edges_removed += block.sparse.edges_removed - old_r

    @torch.no_grad()
    def generate(self, seed_bytes, length=200, temperature=0.8):
        bufs, ptrs = self.init_state()
        generated = list(seed_bytes)
        for i, b in enumerate(seed_bytes):
            logits, bufs, ptrs = self.forward_step(
                torch.tensor(b, dtype=torch.long),
                torch.tensor(i, dtype=torch.long), bufs, ptrs)
        pos = len(seed_bytes)
        for _ in range(length):
            probs = F.softmax(logits / temperature, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            generated.append(next_byte)
            logits, bufs, ptrs = self.forward_step(
                torch.tensor(next_byte, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long), bufs, ptrs)
            pos += 1
        return bytes(generated).decode('utf-8', errors='replace')


# ── Training ──────────────────────────────────────────────────

def train_model(config_name, d_model, n_blocks, buffer_size, max_steps=5000):
    max_edges = d_model * d_model // 2
    model = INSTNCTLM(d_model=d_model, n_blocks=n_blocks,
                       buffer_size=buffer_size, max_edges=max_edges)
    n_params = model.count_params()
    print(f"\n{'='*70}")
    print(f"  {config_name}: {n_params:,} params, d={d_model}, blocks={n_blocks}, buf={buffer_size}")
    print(f"{'='*70}")

    data, _ = build_corpus_bytes()
    data_len = len(data)
    print(f"  Corpus: {data_len:,} bytes", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps)

    bufs, ptrs = model.init_state()
    pos_in_data = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    last_report_added = 0
    last_report_removed = 0
    seq_len = 64
    start_time = time.time()

    results = {'config': config_name, 'params': n_params,
               'steps': [], 'losses': [], 'accs': [],
               'tok_secs': [], 'edges': [], 'texts': []}

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        seq_correct = 0

        for t in range(seq_len):
            idx = (pos_in_data + t) % data_len
            next_idx = (pos_in_data + t + 1) % data_len
            byte_t = data[idx]
            target = data[next_idx]
            logits, bufs_new, ptrs_new = model.forward_step(
                byte_t, torch.tensor(step * seq_len + t, dtype=torch.long),
                bufs, ptrs)
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            total_loss = total_loss + loss
            if logits.argmax().item() == target.item():
                seq_correct += 1
            bufs = [b.detach() for b in bufs_new]
            ptrs = ptrs_new

        avg_loss = total_loss / seq_len
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        pos_in_data = (pos_in_data + seq_len) % data_len
        running_loss += avg_loss.item()
        running_correct += seq_correct
        running_total += seq_len

        model.update_arousal(avg_loss.item())
        if step % 10 == 0:
            model.do_self_wiring()

        if step % 500 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            n_report = min(step, 500)
            avg_l = running_loss / n_report
            acc = running_correct / running_total * 100
            tok_sec = (step * seq_len) / elapsed
            n_edges = model.count_active_edges()
            w_add = model.total_edges_added - last_report_added
            w_rem = model.total_edges_removed - last_report_removed
            last_report_added = model.total_edges_added
            last_report_removed = model.total_edges_removed
            n_a, n_r = model.get_wiring_intensity()

            print(f"  [{step:>5d}/{max_steps}] loss={avg_l:.4f}  acc={acc:>5.1f}%  "
                  f"tok/s={tok_sec:.0f}  edges={n_edges}  "
                  f"wire=+{w_add}/-{w_rem}  conf={model.arousal_confidence:.3f}→add={n_a},rem={n_r}",
                  flush=True)

            seeds = ["The ", "I want to ", "Hello "]
            seed = seeds[((step // 500) - 1) % len(seeds)]
            seed_bytes = list(seed.encode('utf-8'))
            sample = model.generate(seed_bytes, length=200, temperature=0.8)
            print(f"  GEN[{seed!r}]: {sample[:120]!r}", flush=True)

            results['steps'].append(step)
            results['losses'].append(avg_l)
            results['accs'].append(acc)
            results['tok_secs'].append(tok_sec)
            results['edges'].append(n_edges)
            results['texts'].append(sample)

            running_loss = 0.0
            running_correct = 0
            running_total = 0

    results['total_time'] = time.time() - start_time
    print(f"\n  DONE: {results['total_time']:.1f}s", flush=True)
    return results


def main():
    print("VRAXION v23 — INSTNCT CPU Language Learning Experiment")
    print("=" * 70)
    print(f"PyTorch {torch.__version__}, threads={torch.get_num_threads()}")

    all_results = []

    # Tiny: d=128, 4 blocks, buf=256 — fast on CPU
    all_results.append(train_model("Tiny", d_model=128, n_blocks=4,
                                    buffer_size=256, max_steps=5000))

    # Small: d=256, 4 blocks, buf=128 — moderate speed
    # (reduced blocks and buf vs spec for CPU feasibility)
    all_results.append(train_model("Small", d_model=256, n_blocks=4,
                                    buffer_size=128, max_steps=3000))

    # Medium: d=384, 4 blocks, buf=64 — largest feasible on CPU
    all_results.append(train_model("Medium", d_model=384, n_blocks=4,
                                    buffer_size=64, max_steps=2000))

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Params':>10} {'Loss':>8} {'Acc':>7} {'Tok/s':>8} {'Edges':>8} {'Time':>7}")
    print("-" * 72)
    for r in all_results:
        fl = r['losses'][-1] if r['losses'] else float('nan')
        fa = r['accs'][-1] if r['accs'] else 0
        ft = r['tok_secs'][-1] if r['tok_secs'] else 0
        fe = r['edges'][-1] if r['edges'] else 0
        print(f"{r['config']:<20} {r['params']:>10,} {fl:>8.4f} "
              f"{fa:>6.1f}% {ft:>8.0f} {fe:>8} {r['total_time']:>6.1f}s")

    print(f"\n{'='*70}")
    print("\nFINAL GENERATED SAMPLES:")
    print("-" * 70)
    for r in all_results:
        if r['texts']:
            print(f"\n{r['config']}:")
            print(f"  {r['texts'][-1][:200]!r}")

    print("\n\nEVALUATION:")
    print("-" * 70)
    for r in all_results:
        if not r['losses']:
            continue
        fl = r['losses'][-1]
        fa = r['accs'][-1]
        print(f"\n{r['config']} ({r['params']:,} params):")
        print(f"  Final loss: {fl:.4f}, accuracy: {fa:.1f}%")
        if fa > 50:
            print("  → Strong byte prediction. Learning word boundaries and common patterns.")
        elif fa > 30:
            print("  → Moderate learning. Recognizing frequent byte sequences.")
        elif fa > 15:
            print("  → Basic learning. Picking up character frequencies.")
        else:
            print("  → Minimal learning. Needs more steps or tuning.")

    return all_results


if __name__ == '__main__':
    main()
