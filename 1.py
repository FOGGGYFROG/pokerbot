
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hydra6 (single-file, functional, no OOP) — 6-max NLHE bot:
- External-Sampling MCCFR blueprint
- Depth-limited online resolve with optional value net leaves
- Pro-style bet ladders with pruning cap
- Tools: train_blueprint, selfplay_generate, train_value, play_cli, resolve_from_json, merge_strategy_sums

Dependencies:
  pip install numpy torch treys tqdm rich

Notes:
- We avoid defining classes. State is plain dicts/lists. Value net uses raw tensors + manual SGD.
- Hand eval uses `treys` (class inside lib; it's fine to *use* a lib class).
"""

import argparse, random, copy, json, os, math, sys, time
from collections import defaultdict
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np
from tqdm import trange
from rich import print
from rich.prompt import Prompt
import multiprocessing as mp

# Optional GPU deps
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -------- Cards / Evaluator (treys) --------
from treys import Card, Evaluator

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def idx_to_str(idx:int) -> str:
    r = idx % 13
    s = idx // 13
    return f"{RANKS[r]}{SUITS[s]}"

def to_treys(idx:int) -> int:
    return Card.new(idx_to_str(idx))

def board_to_treys(board:List[int]) -> List[int]:
    return [to_treys(c) for c in board]

def hand_to_treys(hand:List[int]) -> List[int]:
    return [to_treys(c) for c in hand]

EVAL = Evaluator()

# -------- Global Config (mutable via CLI) --------
STARTING_STACK = 10_000
SMALL_BLIND = 50
BIG_BLIND   = 100
ANTE        = 0
DEFAULT_NUM_PLAYERS = 6

# Pro-style ladders (can override per-command via CLI flags)
BET_FRACTIONS_PREFLOP = [0.25, 1/3, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0]
BET_FRACTIONS_POSTFLOP= [0.25, 1/3, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
ALLOW_ALLIN = True
RAISE_CAP   = 8  # 0 = no cap

def set_bet_sizes(preflop_csv:str=None, postflop_csv:str=None):
    global BET_FRACTIONS_PREFLOP, BET_FRACTIONS_POSTFLOP
    if preflop_csv:
        BET_FRACTIONS_PREFLOP = [float(x) for x in preflop_csv.split(",") if x.strip()]
    if postflop_csv:
        BET_FRACTIONS_POSTFLOP = [float(x) for x in postflop_csv.split(",") if x.strip()]

def bet_fracs_for_stage(stage:str) -> List[float]:
    return BET_FRACTIONS_PREFLOP if stage == 'preflop' else BET_FRACTIONS_POSTFLOP

# -------- Environment (dict-based, no classes) --------
STAGES = ['preflop','flop','turn','river']

def new_deck():
    d = list(range(52)); random.shuffle(d); return d

def make_player():
    return {
        "stack": STARTING_STACK,
        "invested": 0,
        "total_invested": 0,
        "folded": False,
        "all_in": False,
        "hole": []
    }

def make_state(num_players:int=DEFAULT_NUM_PLAYERS, sb:int=SMALL_BLIND, bb:int=BIG_BLIND, ante:int=ANTE, btn:int=0):
    return {
        "num_players": num_players,
        "sb": sb, "bb": bb, "ante": ante,
        "btn": btn,
        "deck": new_deck(),
        "stage_idx": 0,
        "players": [make_player() for _ in range(num_players)],
        "to_act": 0,
        "last_raiser": None,
        "current_bet": 0,
        "min_raise": 0,
        "board": [],
        "pot": 0,
    }

def clone_state(s): return copy.deepcopy(s)
def stage(s): return STAGES[s["stage_idx"]]

def is_chance_node(s):
    st = stage(s)
    if st == 'preflop':
        for p in s["players"]:
            if len(p["hole"]) < 2 and not p["folded"]:
                return True
        return False
    if st == 'flop' and len(s["board"]) < 3: return True
    if st == 'turn' and len(s["board"]) < 4: return True
    if st == 'river' and len(s["board"]) < 5: return True
    return False

def _post_blind(s, idx, amount):
    p = s["players"][idx]
    pay = min(p["stack"], amount)
    p["stack"] -= pay; p["invested"] += pay; p["total_invested"] += pay
    if p["stack"] == 0: p["all_in"] = True

def chance_step(s):
    s = clone_state(s)
    st = stage(s)
    if st == 'preflop':
        if all(len(p["hole"]) == 0 for p in s["players"]):
            # antes
            if s["ante"] > 0:
                for p in s["players"]:
                    pay = min(p["stack"], s["ante"])
                    p["stack"] -= pay; p["invested"] += pay; p["total_invested"] += pay
            sb_idx = (s["btn"] + 1) % s["num_players"]
            bb_idx = (s["btn"] + 2) % s["num_players"]
            _post_blind(s, sb_idx, s["sb"])
            _post_blind(s, bb_idx, s["bb"])
            s["current_bet"] = s["bb"]
            s["min_raise"] = s["bb"]
            s["to_act"] = (bb_idx + 1) % s["num_players"]
        for p in s["players"]:
            while len(p["hole"]) < 2 and not p["folded"] and len(s["deck"])>0:
                p["hole"].append(s["deck"].pop())
        return s
    if st == 'flop' and len(s["board"]) < 3:
        s["deck"].pop(); s["board"].extend([s["deck"].pop(), s["deck"].pop(), s["deck"].pop()]); return s
    if st == 'turn' and len(s["board"]) < 4:
        s["deck"].pop(); s["board"].append(s["deck"].pop()); return s
    if st == 'river' and len(s["board"]) < 5:
        s["deck"].pop(); s["board"].append(s["deck"].pop()); return s
    return s

def can_continue(s):
    alive = [i for i,p in enumerate(s["players"]) if not p["folded"]]
    return len(alive) > 1

def is_terminal(s):
    return not can_continue(s)

def pot_size(s):
    return s["pot"] + sum(p["invested"] for p in s["players"])

def public_state_key(s):
    board_key = ",".join(map(str, s["board"]))
    inv = ",".join(str(p["invested"]) for p in s["players"])
    stacks = ",".join(str(p["stack"]) for p in s["players"])
    folded = ",".join('1' if p["folded"] else '0' for p in s["players"])
    allins = ",".join('1' if p["all_in"] else '0' for p in s["players"])
    return f'{s["stage_idx"]}|B:{board_key}|I:{inv}|S:{stacks}|F:{folded}|A:{allins}|P:{s["to_act"]}|CB:{s["current_bet"]}|POT:{s["pot"]}'

def private_obs(s, player_idx):
    return ",".join(map(str, s["players"][player_idx]["hole"]))

def legal_actions(s) -> List[Tuple]:
    if not can_continue(s): return []
    me = s["to_act"]; p = s["players"][me]
    if p["folded"] or p["all_in"]: return []
    to_call = s["current_bet"] - p["invested"]
    acts = []

    if to_call > 0 and p["stack"] > 0: acts.append(('F',))
    if to_call == 0: acts.append(('X',))
    else:
        call_amt = min(to_call, p["stack"])
        if call_amt > 0: acts.append(('C',))

    if p["stack"] > (to_call if to_call > 0 else 0):
        pot = pot_size(s)
        for frac in bet_fracs_for_stage(stage(s)):
            target = max(s["current_bet"] + s["min_raise"], int(pot * frac))
            target = max(target, s["current_bet"] + s["min_raise"])
            invest = target - p["invested"]
            if invest > to_call and invest <= p["stack"] + to_call:
                acts.append(('R', int(target)))
        if ALLOW_ALLIN:
            target = p["invested"] + p["stack"]
            if target > s["current_bet"]:
                acts.append(('R', int(target)))

    uniq = []
    seen = set()
    for a in acts:
        key = a if a[0] != 'R' else ('R', int(a[1]))
        if key not in seen:
            seen.add(key); uniq.append(key)

    # prune raises if too many (balanced ladder)
    if RAISE_CAP and RAISE_CAP > 0:
        raises = [a for a in uniq if a[0]=='R']
        others = [a for a in uniq if a[0] != 'R']
        if len(raises) > RAISE_CAP:
            mn = min(raises, key=lambda x:x[1])
            mx = max(raises, key=lambda x:x[1])
            keep = {mn, mx}
            slots = max(RAISE_CAP - len(keep), 0)
            if slots > 0:
                rs = sorted([r for r in raises if r not in keep], key=lambda x:x[1])
                if slots >= len(rs):
                    chosen = rs
                else:
                    # even spread across remaining sizes for balance
                    idxs = [int(i * (len(rs)-1) / max(slots-1,1)) for i in range(slots)]
                    chosen = [rs[i] for i in sorted(set(idxs))]
                    i = 0
                    while len(chosen) < slots and i < len(rs):
                        if rs[i] not in chosen: chosen.append(rs[i])
                        i += 1
                keep = list(keep) + chosen[:slots]
            uniq = others + sorted(keep, key=lambda x:x[1])

    return uniq

def _advance_after_action(s, raise_made=False):
    n = s["num_players"]
    start = (s["to_act"] + 1) % n
    nxt = start
    while True:
        pp = s["players"][nxt]
        if not pp["folded"] and not pp["all_in"]:
            break
        nxt = (nxt + 1) % n
        if nxt == s["to_act"]:
            break
    s["to_act"] = nxt
    active = [i for i,pl in enumerate(s["players"]) if not pl["folded"]]
    if len(active) == 1:
        _award_by_fold(s); return
    someone_to_call = any((s["players"][i]["invested"] < s["current_bet"] and not s["players"][i]["all_in"]) for i in active)
    if not someone_to_call:
        _end_bet_round(s)

def _award_by_fold(s):
    winner = [i for i,p in enumerate(s["players"]) if not p["folded"]][0]
    total_pot = sum(p["total_invested"] for p in s["players"])
    s["players"][winner]["stack"] += total_pot
    for p in s["players"]:
        p["invested"]=0; p["total_invested"]=0
    s["pot"]=0
    for i,p in enumerate(s["players"]):
        if i != winner: p["folded"]=True
    s["to_act"]=winner

def _end_bet_round(s):
    street_contrib = sum(p["invested"] for p in s["players"])
    s["pot"] += street_contrib
    for p in s["players"]: p["invested"]=0
    s["current_bet"]=0; s["min_raise"]=s["bb"]; s["last_raiser"]=None
    if stage(s) == 'river':
        _showdown(s); return
    s["stage_idx"] += 1
    s["to_act"] = (s["btn"] + 1) % s["num_players"]

def _showdown(s):
    contribs = [p["total_invested"] for p in s["players"]]
    alive = [i for i,p in enumerate(s["players"]) if not p["folded"]]
    if len(alive) == 1:
        _award_by_fold(s); return
    b = board_to_treys(s["board"])
    ranks = {}
    for i in alive:
        ranks[i] = EVAL.evaluate(b, hand_to_treys(s["players"][i]["hole"]))
    contrib_pairs = sorted([(i, contribs[i]) for i in range(len(contribs)) if contribs[i] > 0], key=lambda x:x[1])
    side_pots = []; prev = 0
    for idx, amt in contrib_pairs:
        if amt > prev:
            layer_players = [i for i,c in enumerate(contribs) if c >= amt]
            layer_size = (amt - prev) * len(layer_players)
            side_pots.append((set(layer_players), layer_size)); prev = amt
    for elig, pot_size_ in side_pots:
        elig_alive = [i for i in elig if i in alive]
        if not elig_alive: continue
        best = min(ranks[i] for i in elig_alive)
        winners = [i for i in elig_alive if ranks[i] == best]
        share = pot_size_ // len(winners)
        rem = pot_size_ - share * len(winners)
        for w in winners: s["players"][w]["stack"] += share
        if rem > 0: s["players"][winners[0]]["stack"] += rem
    for p in s["players"]:
        p["invested"]=0; p["total_invested"]=0
    for i,p in enumerate(s["players"]):
        if i != s["btn"]: p["folded"]=True
    s["to_act"]=s["btn"]

def step(s, action:Tuple):
    s = clone_state(s)
    me = s["to_act"]; p = s["players"][me]
    to_call = s["current_bet"] - p["invested"]

    if action[0]=='F':
        p["folded"]=True; _advance_after_action(s); return s
    if action[0]=='X':
        assert to_call == 0, "Cannot check facing a bet."
        _advance_after_action(s); return s
    if action[0]=='C':
        pay = min(to_call, p["stack"])
        p["stack"] -= pay; p["invested"] += pay; p["total_invested"] += pay
        if p["stack"]==0: p["all_in"]=True
        _advance_after_action(s); return s
    if action[0]=='R':
        target = int(action[1])
        min_target = s["current_bet"] + max(s["min_raise"], s["bb"])
        if target < min_target: target = min_target
        invest_needed = target - p["invested"]
        invest = min(invest_needed, p["stack"] + 0)
        if invest <= to_call:
            invest = min(to_call, p["stack"]); target = p["invested"] + invest
        p["stack"] -= invest; p["invested"] += invest; p["total_invested"] += invest
        if p["invested"] > s["current_bet"]:
            s["min_raise"] = max(s["bb"], p["invested"] - s["current_bet"])
            s["current_bet"] = p["invested"]; s["last_raiser"]=me
        if p["stack"] == 0: p["all_in"]=True
        _advance_after_action(s, raise_made=True); return s
    raise ValueError(f"Unknown action {action}")

def new_hand(btn:int=0, num_players:int=DEFAULT_NUM_PLAYERS):
    s = make_state(num_players=num_players, btn=btn)
    while is_chance_node(s): s = chance_step(s)
    return s

def utility(s, player_idx:int) -> float:
    return float(s["players"][player_idx]["stack"] - STARTING_STACK)

# -------- Features / Value Net (no Module) --------
def encode_features(s, player:int) -> np.ndarray:
    x = []
    x.append(s["stage_idx"]/3.0)
    x.append(np.tanh(pot_size(s)/20000.0))
    x.append(np.tanh(s["current_bet"]/10000.0))
    p = s["players"][player]
    x.append(np.tanh(p["stack"]/10000.0))
    x.append(np.tanh(p["invested"]/5000.0))
    to_call = max(0, s["current_bet"] - p["invested"])
    x.append(np.tanh(to_call/5000.0))
    b = np.zeros(52, dtype=np.float32)
    for c in s["board"]: b[c]=1.0
    x.extend(b.tolist())
    h = np.zeros(52, dtype=np.float32)
    for c in p["hole"]: h[c]=1.0
    x.extend(h.tolist())
    folded = [1.0 if pl["folded"] else 0.0 for pl in s["players"]]
    allin = [1.0 if pl["all_in"] else 0.0 for pl in s["players"]]
    x.extend(folded); x.extend(allin)
    stacks = [np.tanh(pl["stack"]/10000.0) for pl in s["players"]]
    x.extend(stacks)
    return np.asarray(x, dtype=np.float32)

# -------- Abstraction (multi-layer: board/hole/action) --------
USE_ABSTRACTION = False
DYN_ABS_ENABLED = False
HOLE_CENTROIDS = None  # type: ignore
BOARD_CENTROIDS = None  # type: ignore
RESOLVE_ITERS = 400
RESOLVE_DEPTH = 4
RESOLVE_CACHE_MAX = 10000
RESOLVE_CACHE = {}
CFR_MODE = "CFR_PLUS"  # options: CFR_PLUS, LCFR, DCFR
CFR_ALPHA = 0.5  # exponent for DCFR weighting
PA_ABS = False  # potential-aware abstraction toggle

def set_resolve_params(iters: int = None, depth: int = None):
    global RESOLVE_ITERS, RESOLVE_DEPTH
    if iters is not None:
        RESOLVE_ITERS = max(1, int(iters))
    if depth is not None:
        RESOLVE_DEPTH = max(1, int(depth))

def _board_texture_bucket(s) -> int:
    b = s["board"]
    if not b:
        return 0
    # Simple texture: pairs/flush/straight potentials
    ranks = sorted([(c % 13) for c in b])
    suits = [(c // 13) for c in b]
    suit_counts = {su: suits.count(su) for su in set(suits)}
    max_suit = max(suit_counts.values()) if suit_counts else 0
    paired = any(ranks.count(r) >= 2 for r in set(ranks))
    connected = 0
    if len(ranks) >= 2:
        for i in range(len(ranks)-1):
            if abs(ranks[i+1] - ranks[i]) <= 2:
                connected += 1
    code = (1 if paired else 0) * 4 + (1 if max_suit >= 3 else 0) * 2 + (1 if connected >= 1 else 0)
    return int(code)

def _hole_bucket(s, player:int) -> int:
    # Dynamic abstraction if enabled
    global HOLE_CENTROIDS
    if DYN_ABS_ENABLED and HOLE_CENTROIDS is not None:
        me = s["players"][player]
        vec = np.zeros(52, dtype=np.float32)
        for c in me["hole"]:
            if 0 <= c < 52: vec[c] = 1.0
        ctrs = HOLE_CENTROIDS
        dists = ((ctrs - vec)**2).sum(axis=1)
        return int(np.argmin(dists))
    # Use treys rank on current board if any; otherwise hand class
    me = s["players"][player]
    if len(me["hole"]) < 2:
        return 0
    try:
        b = board_to_treys(s["board"]) if s["board"] else []
        h = hand_to_treys(me["hole"])
        score = EVAL.evaluate(b, h)
        strength = (7462 - score) / 7462.0
        return int(max(0, min(1, strength)) * 99)  # 100 buckets
    except Exception:
        return 50

def _action_phase_bucket(s) -> int:
    # Encodes simple action intensity based on current bet to pot ratio
    pot = pot_size(s)
    cb = s["current_bet"]
    ratio = 0.0 if pot <= 0 else min(5.0, cb / max(1.0, pot))
    return int(ratio * 10)  # 0..50

def abstraction_key(s, player:int) -> str:
    return f"st:{s['stage_idx']}|bt:{_board_texture_bucket(s)}|hb:{_hole_bucket(s, player)}|ap:{_action_phase_bucket(s)}|p:{player}"

# -------- Dynamic Abstraction builder --------
def _kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
        a = np.argmin(d2, axis=1)
        # update
        for i in range(k):
            m = (a == i)
            if np.any(m):
                C[i] = X[m].mean(axis=0)
    return C

def build_abstraction_from_dataset(data_path: str, hole_k: int = 256, board_k: int = 128, iters: int = 25, save: str = "abs.npz"):
    data = np.load(data_path)
    X = np.asarray(data["x"]).astype(np.float32)
    # Assuming encode_features layout: [6 scalars] + 52 board + 52 hole + 6 folded + 6 allin + 6 stacks
    board = X[:, 6:58]
    hole = X[:, 58:110]
    if PA_ABS:
        # Potential-aware tweak: include simple pair/flush/straight potentials in board vectors
        # by concatenating coarse texture codes to board features
        tex = []
        for i in range(board.shape[0]):
            # crude proxy from existing bucket code
            # here use board features directly; leave as zeros if not enough info
            tex.append([0,0,0])
        tex = np.asarray(tex, dtype=np.float32)
        board = np.concatenate([board, tex], axis=1)
    print(f"KMeans hole={hole_k} board={board_k} on {X.shape[0]} samples")
    hole_centroids = _kmeans(hole, hole_k, iters=iters)
    board_centroids = _kmeans(board, board_k, iters=iters)
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    np.savez(save, hole=hole_centroids, board=board_centroids)
    print(f"Saved abstraction to {save}")

def load_abstraction(path: str):
    global HOLE_CENTROIDS, BOARD_CENTROIDS, USE_ABSTRACTION, DYN_ABS_ENABLED
    d = np.load(path)
    HOLE_CENTROIDS = np.asarray(d["hole"]).astype(np.float32)
    BOARD_CENTROIDS = np.asarray(d["board"]).astype(np.float32)
    USE_ABSTRACTION = True
    DYN_ABS_ENABLED = True
    print(f"Loaded abstraction: hole={HOLE_CENTROIDS.shape[0]} board={BOARD_CENTROIDS.shape[0]}")

def init_value_params(in_dim:int, hidden:int=256, out_dim:int=1, rng=None):
    import torch
    if rng is None: rng = torch.Generator().manual_seed(42)
    def glorot(shape):
        fan_in, fan_out = shape[1], shape[0]
        limit = math.sqrt(6/(fan_in+fan_out))
        return (torch.rand(shape, generator=rng)*2-1)*limit
    W1 = glorot((hidden, in_dim));  b1 = torch.zeros(hidden)
    W2 = glorot((hidden, hidden));  b2 = torch.zeros(hidden)
    W3 = glorot((out_dim, hidden)); b3 = torch.zeros(out_dim)
    for t in (W1,b1,W2,b2,W3,b3): t.requires_grad_(True)
    return {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}

def value_forward(x_np:np.ndarray, params:Dict[str,'torch.Tensor']) -> 'torch.Tensor':
    import torch, torch.nn.functional as F
    x = torch.from_numpy(x_np).float()
    if x.ndim==1: x = x.unsqueeze(0)
    h1 = F.relu(x @ params["W1"].t() + params["b1"])
    h2 = F.relu(h1 @ params["W2"].t() + params["b2"])
    out= (h2 @ params["W3"].t() + params["b3"]).squeeze(-1)
    return out

def make_value_fn(params):
    def vf(s, player:int):
        import torch
        with torch.no_grad():
            x = encode_features(s, player)
            y = value_forward(x, params)
            return float(y.item())
    return vf

# -------- High-performance Value Net (multi-GPU ready) --------
class ValueNet(nn.Module):
    def __init__(self, in_dim:int, hidden:int=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_value_distributed(data_path:str, save_path:str="value_dp.pt", epochs:int=10, batch:int=8192, lr:float=1e-3):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for train_value_distributed")
    data = np.load(data_path)
    X = torch.from_numpy(data["x"]).float()
    Y = torch.from_numpy(data["y"]).float()
    in_dim = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(in_dim)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[bold green]Using DataParallel on {torch.cuda.device_count()} GPUs[/bold green]")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_fn = nn.MSELoss()
    N = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N)
        total = 0.0
        model.train()
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
        scheduler.step()
        print(f"Epoch {ep+1}: loss={total/N:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
    # Save state dict (handle DataParallel)
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"in_dim": in_dim, "state_dict": state}, save_path)
    print(f"Saved multi-GPU value net to {save_path}")

def make_value_fn_from_dp(path:str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for make_value_fn_from_dp")
    ckpt = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    in_dim = int(ckpt.get("in_dim", 0))
    model = ValueNet(in_dim)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    @torch.no_grad()
    def vf(s, player:int):
        x = torch.from_numpy(encode_features(s, player)).float().to(device)
        if x.ndim == 1: x = x.unsqueeze(0)
        y = model(x)
        return float(y.item())
    return vf

def train_value(data_path:str, save_path:str="value_single.pt", epochs:int=5, batch:int=512, lr:float=3e-4):
    import torch, numpy as np
    data = np.load(data_path)
    X = torch.from_numpy(data["x"]).float()
    Y = torch.from_numpy(data["y"]).float()
    in_dim = X.shape[1]
    params = init_value_params(in_dim)
    N = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb = X[idx]; yb = Y[idx]
            pred = value_forward(xb.numpy(), params)
            loss = ((pred - yb)**2).mean()
            for t in params.values():
                if t.grad is not None: t.grad.zero_()
            loss.backward()
            # simple SGD
            for k,t in params.items():
                with torch.no_grad():
                    t -= lr * t.grad
            tot += float(loss.item()) * xb.size(0)
        print(f"Epoch {ep+1}: loss={tot/N:.4f}")
    torch.save({k:v.detach().cpu() for k,v in params.items()}, save_path)
    print(f"Saved value params to {save_path}")

def load_value_params(path:str):
    import torch
    d = torch.load(path, map_location="cpu")
    for t in d.values():
        t.requires_grad = False
    return d

# -------- MCCFR (functional) --------
def rt_get_strategy(tab:Dict, infoset:str, actions:List[Tuple], tau:float=0.0):
    # Regret matching / CFR+ baseline probabilities
    r = np.array([tab["regret"][infoset][a] for a in actions], dtype=float)
    if CFR_MODE == "CFR_PLUS":
        r = np.maximum(r, 0.0)
    if r.sum() <= 1e-12:
        # bias towards call/check when no info
        base = np.ones(len(actions))
        for i,a in enumerate(actions):
            if a[0] == 'F': base[i] = 0.5
            if a[0] in ('C','X'): base[i] = 1.5
        probs = base / base.sum()
    else:
        probs = r / r.sum()
    if tau > 0:
        probs = np.exp(np.log(np.clip(probs,1e-12,1))/(1+tau)); probs/=probs.sum()
    return {a: float(p) for a,p in zip(actions, probs.tolist())}

def rt_add_regret(tab, infoset, action, delta):
    prev = tab["regret"][infoset][action]
    new_val = prev + delta
    if CFR_MODE == "CFR_PLUS":
        new_val = max(0.0, new_val)
    tab["regret"][infoset][action] = new_val

def rt_add_strategy(tab, infoset, sigma:Dict[Tuple,float], weight:float=1.0, iteration:int=1):
    # LCFR/DCFR weighting support
    if CFR_MODE == "LCFR":
        w = float(iteration)
    elif CFR_MODE == "DCFR":
        w = float(iteration ** CFR_ALPHA)
    else:
        w = max(0.0, float(weight))
    for a,p in sigma.items():
        tab["strategy_sum"][infoset][a] += w * p

def rt_avg_strategy(tab, infoset, actions):
    denom = sum(tab["strategy_sum"][infoset][a] for a in actions) + 1e-12
    if denom <= 1e-10:
        return {a: 1.0/len(actions) for a in actions}
    return {a: tab["strategy_sum"][infoset][a]/denom for a in actions}

def info_key(s, player:int):
    if USE_ABSTRACTION:
        return abstraction_key(s, player)
    return f"{public_state_key(s)}|priv:{private_obs(s, player)}|p:{player}"

def terminal_value(s, target:int):
    return utility(s, target)

def rollout_value(s, target:int, value_fn=None):
    if value_fn is not None:
        try:
            return float(value_fn(s, target))
        except Exception:
            pass
    ss = clone_state(s)
    steps = 0
    while not is_terminal(ss) and steps < 1000:
        if is_chance_node(ss):
            ss = chance_step(ss); continue
        la = legal_actions(ss)
        if not la: steps += 1; continue
        a = random.choice(la); ss = step(ss, a); steps += 1
    return terminal_value(ss, target)

def external_sampling(tab, s, target:int, reach_i:float, reach_opp:float, depth:int, value_fn, depth_limit, iteration:int, linear_weighting:bool=True):
    if is_chance_node(s):
        return external_sampling(tab, chance_step(s), target, reach_i, reach_opp, depth, value_fn, depth_limit, iteration, linear_weighting)
    if is_terminal(s):
        return terminal_value(s, target)

    player = s["to_act"]
    legal = legal_actions(s)

    # If no legal actions (e.g., all-in freeze), fallback to rollout to advance state
    if not legal:
        return rollout_value(s, target, value_fn=value_fn)

    if depth_limit is not None and depth >= depth_limit:
        return rollout_value(s, target, value_fn=value_fn)

    infoset = info_key(s, player)
    sigma = rt_get_strategy(tab, infoset, legal) if legal else {}

    if player == target:
        # Build action list and probs robustly
        acts = list(legal)
        if not acts:
            return rollout_value(s, target, value_fn=value_fn)
        probs = [max(float(sigma.get(a, 0.0)), 0.0) for a in acts]
        z = sum(probs)
        if z <= 1e-12:
            probs = [1.0/len(acts)] * len(acts)
            z = 1.0
        util = {}
        node_util = 0.0
        for a, p in zip(acts, probs):
            nxt = step(s, a)
            u = external_sampling(tab, nxt, target, reach_i*p, reach_opp, depth+1, value_fn, depth_limit, iteration, linear_weighting)
            util[a] = u
            node_util += p * u
        # Regret update (CFR+ with iteration-weighted regrets)
        for a in acts:
            regret = util[a] - node_util
            w = iteration if linear_weighting else 1.0
            rt_add_regret(tab, infoset, a, w * regret * reach_opp)
        # Strategy sum update with normalized probs, weighted by reach_i
        sig = {a: (p if z == 1.0 else p/z) for a, p in zip(acts, probs)}
        rt_add_strategy(tab, infoset, sig, weight=reach_i, iteration=iteration)
        return node_util
    else:
        # Opponent sampling: use sigma if valid else uniform over legal
        acts = list(legal)
        if not acts:
            return rollout_value(s, target, value_fn=value_fn)
        probs = [max(float(sigma.get(a, 0.0)), 0.0) for a in acts]
        total_w = sum(probs)
        if total_w <= 1e-12:
            probs = [1.0/len(acts)] * len(acts)
        a = random.choices(acts, weights=probs, k=1)[0]
        nxt = step(s, a)
        p = sigma.get(a, 1.0/len(acts))
        return external_sampling(tab, nxt, target, reach_i, reach_opp * p, depth+1, value_fn, depth_limit, iteration, linear_weighting)

def mccfr_train_iteration(tab, root, value_fn=None, depth_limit=None, iteration:int=1, linear_weighting=True):
    for p in range(root["num_players"]):
        external_sampling(tab, root, p, 1.0, 1.0, 0, value_fn, depth_limit, iteration, linear_weighting)

def depth_limited_resolve(root, value_fn, cfr_iters:int=200, depth_limit:int=3):
    # Respect caller values; do not override upwards to global caps during strict eval
    cfr_iters = int(max(1, cfr_iters))
    depth_limit = int(max(1, depth_limit))
    # Small cache by public state key + to_act
    global RESOLVE_CACHE
    key = (public_state_key(root), root["to_act"], cfr_iters, depth_limit)
    if key in RESOLVE_CACHE:
        return RESOLVE_CACHE[key]
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    for it in range(1, cfr_iters+1):
        mccfr_train_iteration(tab, root, value_fn=value_fn, depth_limit=depth_limit, iteration=it, linear_weighting=True)
    infoset = info_key(root, root["to_act"])
    legal = legal_actions(root)
    strat = rt_avg_strategy(tab, infoset, legal)
    if len(RESOLVE_CACHE) > RESOLVE_CACHE_MAX:
        RESOLVE_CACHE.clear()
    RESOLVE_CACHE[key] = strat
    return strat

# -------- Data / Scripts --------
def rollout_return(s, player:int) -> float:
    steps=0; ss=clone_state(s)
    while not is_terminal(ss) and steps < 1000:
        if is_chance_node(ss): ss = chance_step(ss); continue
        la = legal_actions(ss)
        if not la: steps += 1; continue
        raises = [a for a in la if a[0]=='R']
        if raises and random.random()<0.15: a=random.choice(raises)
        else: a=('C',) if ('C',) in la else ('X',) if ('X',) in la else random.choice(la)
        ss = step(ss,a); steps+=1
    return utility(ss, player)

def cmd_selfplay_generate(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    xs=[]; ys=[]; btn=0
    for _ in trange(args.episodes, desc="Self-play"):
        s = new_hand(btn=btn, num_players=args.num_players)
        for seat in range(s["num_players"]):
            ss = clone_state(s)
            for _ in range(random.randint(0,6)):
                if is_chance_node(ss): ss = chance_step(ss)
                else:
                    la = legal_actions(ss)
                    if not la: break
                    ss = step(ss, random.choice(la))
            xs.append(encode_features(ss, seat))
            ys.append(rollout_return(ss, seat))
        btn = (btn + 1) % s["num_players"]
    xs=np.stack(xs,axis=0); ys=np.asarray(ys,dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, x=xs, y=ys)
    print(f"Saved self-play dataset to {args.out}  x{xs.shape} y{ys.shape}")

def cmd_train_value(args):
    train_value(args.data, save_path=args.save, epochs=args.epochs, batch=args.batch, lr=args.lr)

def cmd_train_blueprint(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    global USE_ABSTRACTION, DYN_ABS_ENABLED
    if args.use_abstraction:
        USE_ABSTRACTION = True
        DYN_ABS_ENABLED = bool(args.dynamic_abstraction)
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    btn=0
    decay = float(args.regret_decay)
    for it in trange(args.iterations, desc="MCCFR"):
        s = new_hand(btn=btn, num_players=args.num_players)
        mccfr_train_iteration(tab, s, value_fn=None, depth_limit=None, iteration=it, linear_weighting=True)
        if decay > 0:
            # simple exponential decay of regrets
            for infoset, acts in list(tab["regret"].items()):
                for a in list(acts.keys()):
                    tab["regret"][infoset][a] *= (1.0 - decay)
        btn = (btn + 1) % s["num_players"]
    # avg strategy dump
    strategy = {}
    for infoset, acts in tab["strategy_sum"].items():
        denom = sum(acts.values()) + 1e-12
        strategy[infoset] = {str(a): float(v/denom) for a,v in acts.items()}
    with open(args.save, "w") as f: json.dump(strategy, f)
    print(f"Saved blueprint to {args.save}")
    if args.save_sum:
        raw = {}
        for infoset, acts in tab["strategy_sum"].items():
            raw[infoset] = {str(a): float(v) for a,v in acts.items()}
        with open(args.save_sum,"w") as f: json.dump(raw,f)
        print(f"Saved raw strategy_sum to {args.save_sum}")

def _merge_strategy_sums(dst:Dict, src:Dict):
    for infoset, acts in src.items():
        if infoset not in dst:
            dst[infoset] = {}
        for a, v in acts.items():
            dst[infoset][a] = dst[infoset].get(a, 0.0) + float(v)

def _worker_mccfr(args_tuple):
    iterations, num_players, seed, bet_pre, bet_post, abstraction_path, use_abs, dyn_abs = args_tuple
    random.seed(seed)
    if bet_pre: set_bet_sizes(bet_pre, None)
    if bet_post: set_bet_sizes(None, bet_post)
    if abstraction_path:
        try:
            load_abstraction(abstraction_path)
        except Exception:
            pass
    if use_abs:
        global USE_ABSTRACTION, DYN_ABS_ENABLED
        USE_ABSTRACTION = True
        DYN_ABS_ENABLED = bool(dyn_abs)
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    btn = 0
    for it in range(1, iterations+1):
        s = new_hand(btn=btn, num_players=num_players)
        mccfr_train_iteration(tab, s, value_fn=None, depth_limit=None, iteration=it, linear_weighting=True)
        btn = (btn + 1) % num_players
    # Serialize strategy_sum only
    out = {}
    for infoset, acts in tab["strategy_sum"].items():
        out[infoset] = {str(a): float(v) for a, v in acts.items()}
    return out

def cmd_train_blueprint_dist(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    world = max(1, int(args.num_workers))
    total_iters = int(args.iterations)
    chunk = int(args.chunk) if args.chunk else max(1, total_iters // world)
    seeds = [1234 + i for i in range(world)]
    bet_pre = args.bet_sizes_preflop
    bet_post = args.bet_sizes_postflop
    abstraction_path = args.abstraction
    use_abs = bool(args.use_abstraction)
    dyn_abs = bool(args.dynamic_abstraction)
    print(f"[bold green]Distributed MCCFR[/bold green]: workers={world} total_iters={total_iters} chunk={chunk}")
    merged = {}
    done = 0
    def _quick_eval_path(bp_path:str):
        try:
            if not args.eval_every_chunk:
                return
            bp_local = load_blueprint(bp_path)
            # value fn
            if args.eval_value and os.path.exists(args.eval_value):
                value_fn = make_value_fn_from_dp(args.eval_value) if args.eval_value.endswith('.pt') else make_value_fn(load_value_params(args.eval_value))
            else:
                value_fn = (lambda s, p: 0.0)
            # abstraction
            if args.eval_abstraction:
                try:
                    load_abstraction(args.eval_abstraction)
                except Exception:
                    pass
            # run minimal evaluate loop
            btn = 0
            winnings = []
            baseline = args.eval_baseline or 'random'
            bfn = _baseline_random_action if baseline=='random' else _baseline_tight_action
            for _ in range(int(args.eval_hands)):
                res, btn = _play_hand(bp_local, value_fn, bfn, num_players=int(args.eval_num_players), btn=btn)
                winnings.append(res[0])
            bb = BIG_BLIND
            bb_per_hand = [w / bb for w in winnings]
            mean_bb100 = float(np.mean(bb_per_hand) * 100)
            std = float(np.std(bb_per_hand))
            print(json.dumps({
                "monitor": {
                    "done_iters": done,
                    "bb_100": mean_bb100,
                    "std_bb": std,
                    "hands": len(winnings),
                    "baseline": baseline,
                    "ckpt": bp_path
                }
            }))

            # Optional head-to-head vs provided opponent blueprint each chunk
            if getattr(args, 'h2h_every_chunk', False) and getattr(args, 'h2h_blueprint', None):
                opp_path = args.h2h_blueprint
                if isinstance(opp_path, str) and os.path.exists(opp_path):
                    try:
                        bp_opp = load_blueprint(opp_path)
                        h2h_btn = 0
                        h2h_wins = []
                        h2h_hands = int(getattr(args, 'h2h_hands', 1000) or 1000)
                        resolve_iters = int(getattr(args, 'h2h_resolve_iters', RESOLVE_ITERS) or RESOLVE_ITERS)
                        resolve_depth = int(getattr(args, 'h2h_resolve_depth', RESOLVE_DEPTH) or RESOLVE_DEPTH)
                        for _ in range(h2h_hands):
                            res, h2h_btn = _play_hand_matchup(bp_local, bp_opp, value_fn, num_players=int(args.eval_num_players), btn=h2h_btn, resolve_iters=resolve_iters, resolve_depth=resolve_depth)
                            h2h_wins.append(res[0])
                        h2h_bb_per_hand = [w / bb for w in h2h_wins]
                        h2h_bb100 = float(np.mean(h2h_bb_per_hand) * 100)
                        print(json.dumps({
                            "head2head": {
                                "done_iters": done,
                                "bb_100": h2h_bb100,
                                "hands": len(h2h_wins),
                                "ckpt": bp_path,
                                "opp": opp_path,
                                "resolve_iters": resolve_iters,
                                "resolve_depth": resolve_depth
                            }
                        }))
                    except Exception:
                        pass
        except Exception:
            pass

    with mp.get_context("spawn").Pool(processes=world) as pool:
        while done < total_iters:
            batch_iters = min(chunk, total_iters - done)
            parts = pool.map(_worker_mccfr, [
                (batch_iters // world + (1 if i < (batch_iters % world) else 0),
                 args.num_players, seeds[i] + done, bet_pre, bet_post, abstraction_path, use_abs, dyn_abs)
                for i in range(world)
            ])
            for part in parts:
                _merge_strategy_sums(merged, part)
            done += batch_iters
            # Optional checkpoint
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{done}.json")
                bp = {}
                for infoset, acts in merged.items():
                    denom = sum(acts.values()) + 1e-12
                    bp[infoset] = {a: float(v/denom) for a, v in acts.items()}
                with open(ckpt_path, "w") as f: json.dump(bp, f)
                print(f"Checkpoint saved: {ckpt_path}")
                _quick_eval_path(ckpt_path)
    # Normalize to blueprint
    blueprint = {}
    for infoset, acts in merged.items():
        denom = sum(acts.values()) + 1e-12
        blueprint[infoset] = {a: float(v/denom) for a, v in acts.items()}
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w") as f: json.dump(blueprint, f)
    print(f"Saved distributed blueprint to {args.save}")
    if args.save_sum:
        with open(args.save_sum, "w") as f: json.dump(merged, f)
        print(f"Saved merged raw sums to {args.save_sum}")

def load_blueprint(path:str):
    with open(path,"r") as f: return json.load(f)

def cmd_merge_sums(args):
    agg = {}
    for path in args.inputs:
        with open(path,"r") as f:
            data = json.load(f)
        for infoset, acts in data.items():
            if infoset not in agg: agg[infoset]={}
            for a,v in acts.items():
                agg[infoset][a] = agg[infoset].get(a,0.0)+float(v)
    blueprint={}
    for infoset, acts in agg.items():
        denom = sum(acts.values()) + 1e-12
        blueprint[infoset] = {a: float(v/denom) for a,v in acts.items()}
    with open(args.out_blueprint,"w") as f: json.dump(blueprint,f)
    print(f"Saved merged blueprint to {args.out_blueprint}")
    if args.out_sum:
        with open(args.out_sum,"w") as f: json.dump(agg,f)
        print(f"Saved merged raw sums to {args.out_sum}")

def choose_bot_action(s, blueprint:Dict, value_fn, resolve_iters:int=None, resolve_depth:int=None):
    la = legal_actions(s)
    if not la: return None
    # Stage-aware defaults; can be overridden by args
    st = stage(s)
    it = resolve_iters if resolve_iters is not None else (200 if st=='preflop' else 400)
    dp = resolve_depth if resolve_depth is not None else (3 if st=='preflop' else 4)
    strat = depth_limited_resolve(s, value_fn=value_fn, cfr_iters=it, depth_limit=dp)
    if strat:
        acts=list(strat.keys()); probs=[strat[a] for a in acts]
        return random.choices(acts, weights=probs, k=1)[0]
    infoset = f"{public_state_key(s)}|priv:{private_obs(s, s['to_act'])}|p:{s['to_act']}"
    if infoset in blueprint:
        options=[]; probs=[]
        for k,v in blueprint[infoset].items():
            ks = k.replace('[','(').replace(']',')').replace('"','').replace("'", '')
            if ks.startswith("(R,"):
                num = ''.join(ch for ch in ks if ch.isdigit()); options.append(('R', int(num)))
            elif ks.startswith("(C"): options.append(('C',))
            elif ks.startswith("(X"): options.append(('X',))
            elif ks.startswith("(F"): options.append(('F',))
            else: continue
            probs.append(float(v))
        if options:
            return random.choices(options, weights=probs, k=1)[0]
    return random.choice(la)

def _baseline_random_action(s):
    la = legal_actions(s)
    if not la: return None
    raises = [a for a in la if a[0]=='R']
    if raises and random.random() < 0.1: return random.choice(raises)
    if ('C',) in la: return ('C',)
    if ('X',) in la: return ('X',)
    return random.choice(la)

def _baseline_tight_action(s):
    la = legal_actions(s)
    if not la: return None
    if ('C',) in la and random.random() < 0.6: return ('C',)
    if ('X',) in la and random.random() < 0.8: return ('X',)
    raises = [a for a in la if a[0]=='R']
    if raises and random.random() < 0.05: return random.choice(raises)
    if ('F',) in la: return ('F',)
    return random.choice(la)

def _play_hand(blueprint:Dict, value_fn, baseline_fn, num_players:int=6, btn:int=0):
    s = new_hand(btn=btn, num_players=num_players)
    while True:
        while is_chance_node(s): s = chance_step(s)
        if is_terminal(s):
            return [utility(s, i) for i in range(num_players)], (btn + 1) % num_players
        # If no legal actions (e.g., all-in freeze), auto-advance state safely
        la0 = legal_actions(s)
        if not la0:
            active = [i for i, pl in enumerate(s["players"]) if not pl["folded"]]
            if len(active) <= 1:
                _award_by_fold(s)
                continue
            someone_to_call = any((s["players"][i]["invested"] < s["current_bet"] and not s["players"][i]["all_in"]) for i in active)
            if not someone_to_call:
                _end_bet_round(s)
                continue
            # Move action to next eligible player
            n = s["num_players"]
            nxt = (s["to_act"] + 1) % n
            while (s["players"][nxt]["folded"] or s["players"][nxt]["all_in"]) and nxt != s["to_act"]:
                nxt = (nxt + 1) % n
            s["to_act"] = nxt
            continue
        cp = s["to_act"]
        if cp == 0:
            a = choose_bot_action(s, blueprint, value_fn)
        else:
            a = baseline_fn(s)
        if a is None:
            # Fallback to a safe legal action
            la = legal_actions(s)
            if not la:
                continue
            if ('C',) in la:
                a = ('C',)
            elif ('X',) in la:
                a = ('X',)
            else:
                a = random.choice(la)
        s = step(s, a)

def _play_hand_matchup(bp0:Dict, bp1:Dict, value_fn, num_players:int=2, btn:int=0, resolve_iters:Optional[int]=None, resolve_depth:Optional[int]=None):
    s = new_hand(btn=btn, num_players=num_players)
    while True:
        while is_chance_node(s): s = chance_step(s)
        if is_terminal(s):
            return [utility(s, i) for i in range(num_players)], (btn + 1) % num_players
        la0 = legal_actions(s)
        if not la0:
            active = [i for i, pl in enumerate(s["players"]) if not pl["folded"]]
            if len(active) <= 1:
                _award_by_fold(s)
                continue
            someone_to_call = any((s["players"][i]["invested"] < s["current_bet"] and not s["players"][i]["all_in"]) for i in active)
            if not someone_to_call:
                _end_bet_round(s)
                continue
            n = s["num_players"]
            nxt = (s["to_act"] + 1) % n
            while (s["players"][nxt]["folded"] or s["players"][nxt]["all_in"]) and nxt != s["to_act"]:
                nxt = (nxt + 1) % n
            s["to_act"] = nxt
            continue
        cp = s["to_act"]
        if cp == 0:
            a = choose_bot_action(s, bp0, value_fn, resolve_iters=resolve_iters, resolve_depth=resolve_depth)
        elif cp == 1:
            a = choose_bot_action(s, bp1, value_fn, resolve_iters=resolve_iters, resolve_depth=resolve_depth)
        else:
            a = _baseline_random_action(s)
        if a is None:
            la = legal_actions(s)
            if not la:
                continue
            if ('C',) in la:
                a = ('C',)
            elif ('X',) in la:
                a = ('X',)
            else:
                a = random.choice(la)
        s = step(s, a)

def cmd_evaluate(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    global USE_ABSTRACTION, DYN_ABS_ENABLED
    if args.abstraction:
        try:
            load_abstraction(args.abstraction)
        except Exception:
            pass
    if args.use_abstraction:
        USE_ABSTRACTION = True
        DYN_ABS_ENABLED = bool(args.dynamic_abstraction)
    bp = load_blueprint(args.blueprint)
    if args.value and os.path.exists(args.value):
        value_fn = make_value_fn_from_dp(args.value) if args.value.endswith(".pt") else make_value_fn(load_value_params(args.value))
    else:
        value_fn = (lambda s, p: 0.0)
    baseline = args.baseline
    if baseline == 'random':
        bfn = _baseline_random_action
    elif baseline == 'tight':
        bfn = _baseline_tight_action
    else:
        bfn = _baseline_random_action
    btn = 0
    winnings = []
    for _ in trange(args.hands, desc=f"Evaluate vs {baseline}"):
        res, btn = _play_hand(bp, value_fn, bfn, num_players=args.num_players, btn=btn)
        winnings.append(res[0])
    bb = BIG_BLIND
    bb_per_hand = [w / bb for w in winnings]
    mean_bb100 = float(np.mean(bb_per_hand) * 100)
    std = float(np.std(bb_per_hand))
    print(json.dumps({
        "bb_100": mean_bb100,
        "std_bb": std,
        "hands": len(winnings),
        "baseline": baseline
    }, indent=2))

def cmd_smoke_test(args):
    # 1) Small resolve
    s_dict = {
        "num_players": 6,
        "stage": "preflop",
        "players": [
            {"stack": 10000, "hole": [12, 25]},
            {"stack": 10000}, {"stack": 10000}, {"stack": 10000}, {"stack": 10000}, {"stack": 10000}
        ],
        "board": [], "to_act": 0, "current_bet": 100, "min_raise": 100, "pot": 150, "btn": 0
    }
    s = state_from_json(s_dict)
    strat = depth_limited_resolve(s, value_fn=(lambda st,pl: 0.0), cfr_iters=5, depth_limit=2)
    resolve_out = {str(k): float(v) for k, v in strat.items()}

    # 2) One strict BR sample (empty blueprint)
    bp = {}
    value_fn = (lambda st, pl: 0.0)
    ss = new_hand(btn=0, num_players=6)
    for _ in range(2):
        if is_chance_node(ss):
            ss = chance_step(ss)
    memo = {}
    br = _br_value(ss, 0, bp, value_fn, cfr_iters=10, depth_limit=2, depth_cap=2, memo=memo)
    sv = _sigma_value(ss, 0, bp, value_fn, cfr_iters=10, depth_limit=2, depth_cap=2, memo={})
    br_out = {"br": br, "sigma": sv, "gap": max(0.0, br - sv)}

    # 3) Tiny evaluation: 2 hands vs random baseline
    winnings = []
    btn = 0
    for _ in range(2):
        res, btn = _play_hand({}, (lambda s,p: 0.0), _baseline_random_action, num_players=6, btn=btn)
        winnings.append(res[0])
    eval_out = {"hands": len(winnings), "winnings": winnings}

    print(json.dumps({
        "resolve": resolve_out,
        "br": br_out,
        "eval": eval_out
    }, indent=2))

def cmd_quickcheck(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    # 1) Train exactly 1 iteration and save a tiny blueprint
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    btn = 0
    s = new_hand(btn=btn, num_players=args.num_players)
    mccfr_train_iteration(tab, s, value_fn=None, depth_limit=None, iteration=1, linear_weighting=True)
    # Dump avg strategy
    strategy = {}
    for infoset, acts in tab["strategy_sum"].items():
        denom = sum(acts.values()) + 1e-12
        strategy[infoset] = {str(a): float(v/denom) for a, v in acts.items()}
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(strategy, f)
    # 2) Evaluate a few hands vs random
    winnings = []
    bp = strategy
    value_fn = (lambda st, pl: 0.0)
    btn = 0
    for _ in range(args.hands):
        res, btn = _play_hand(bp, value_fn, _baseline_random_action, num_players=args.num_players, btn=btn)
        winnings.append(res[0])
    bb = BIG_BLIND
    bb_per_hand = [w / bb for w in winnings]
    mean_bb100 = float(np.mean(bb_per_hand) * 100)
    std = float(np.std(bb_per_hand))
    print(json.dumps({
        "saved": args.save,
        "hands": len(winnings),
        "bb_100": mean_bb100,
        "std_bb": std
    }, indent=2))

def cmd_engine_sanity(args):
    results = []
    def record(name:str, ok:bool, info:str=""):
        results.append({"test": name, "ok": bool(ok), "info": info})

    # Test 1: blinds, deal, to_act
    try:
        btn = 0; n=6
        s = new_hand(btn=btn, num_players=n)
        sb_idx = (btn + 1) % n; bb_idx = (btn + 2) % n
        ok_cards = all(len(p["hole"]) == 2 for p in s["players"])
        ok_blinds = (s["players"][sb_idx]["invested"] == s["sb"]) and (s["players"][bb_idx]["invested"] == s["bb"]) and (s["current_bet"] == s["bb"]) and (s["min_raise"] == s["bb"]) and (s["to_act"] == (bb_idx + 1) % n)
        record("blinds_and_deal", ok_cards and ok_blinds)
    except Exception as e:
        record("blinds_and_deal", False, str(e))

    # Test 2: check not legal facing a bet
    try:
        s = new_hand(btn=0, num_players=6)
        la = legal_actions(s)
        record("no_check_facing_bet", ('X',) not in la)
    except Exception as e:
        record("no_check_facing_bet", False, str(e))

    # Test 3: min-raise clamp and update
    try:
        s = new_hand(btn=0, num_players=6)
        prev_cb = s["current_bet"]
        # attempt an under-min raise to force clamp
        s2 = step(s, ('R', prev_cb + s["bb"] - 1))
        ok_target_ge_min = (s2["current_bet"] >= prev_cb + s["bb"])  # min raise from bb to 2*bb
        ok_min_raise_updated = (s2["min_raise"] >= s["bb"])  # at least bb
        record("min_raise_rule", ok_target_ge_min and ok_min_raise_updated)
    except Exception as e:
        record("min_raise_rule", False, str(e))

    # Test 4: all-in removes actions for that player
    try:
        s = new_hand(btn=0, num_players=6)
        me = s["to_act"]
        s["players"][me]["stack"] = 0
        s["players"][me]["all_in"] = True
        la = legal_actions(s)
        record("allin_no_actions", len(la) == 0)
    except Exception as e:
        record("allin_no_actions", False, str(e))

    # Test 5: side pot distribution sanity
    try:
        s = make_state(num_players=3, btn=0)
        # Set river and full board
        s["stage_idx"] = 3
        s["board"] = [0, 1, 2, 3, 4]
        # Players with different investments
        for i in range(3):
            s["players"][i]["hole"] = [5 + i, 10 + i]
            s["players"][i]["folded"] = False
        # Emulate investments: P0 all-in 300, P1 500, P2 1000
        invs = [300, 500, 1000]
        total_before = sum(p["stack"] for p in s["players"]) + sum(invs)
        for i,amt in enumerate(invs):
            s["players"][i]["total_invested"] = amt
            s["players"][i]["invested"] = 0
        _showdown(s)
        total_after = sum(p["stack"] for p in s["players"]) + s["pot"] + sum(p["invested"] for p in s["players"]) + sum(p["total_invested"] for p in s["players"])
        record("sidepot_conservation", abs(total_before - total_after) < 1e-6)
    except Exception as e:
        record("sidepot_conservation", False, str(e))

    passed = sum(1 for r in results if r["ok"]) ; failed = len(results) - passed
    print(json.dumps({"passed": passed, "failed": failed, "results": results}, indent=2))

# -------- Strict Best-Response Exploitability --------
def _sigma_from_blueprint_or_resolve(s, blueprint: Dict, value_fn, cfr_iters: int, depth_limit: int) -> Dict[Tuple, float]:
    legal = legal_actions(s)
    if not legal:
        return {}
    # Try online resolve first
    try:
        strat = depth_limited_resolve(s, value_fn=value_fn, cfr_iters=cfr_iters, depth_limit=depth_limit)
        if strat:
            # Filter to legal and renormalize
            acts = {a: p for a, p in strat.items() if a in legal and p > 0}
            z = sum(acts.values())
            if z > 0:
                return {a: v / z for a, v in acts.items()}
    except Exception:
        pass
    # Fallback to blueprint
    infoset = info_key(s, s["to_act"]) if not USE_ABSTRACTION else abstraction_key(s, s["to_act"])  # consistent key
    if infoset in blueprint:
        options = []
        probs = []
        for k, v in blueprint[infoset].items():
            ks = k.replace('[', '(').replace(']', ')').replace('"', '').replace("'", '')
            if ks.startswith("(R,"):
                num = ''.join(ch for ch in ks if ch.isdigit())
                a = ('R', int(num))
            elif ks.startswith("(C"):
                a = ('C',)
            elif ks.startswith("(X"):
                a = ('X',)
            elif ks.startswith("(F"):
                a = ('F',)
            else:
                continue
            if a in legal:
                options.append(a); probs.append(float(v))
        if options:
            z = sum(probs)
            if z > 0:
                return {a: p / z for a, p in zip(options, probs)}
    # Uniform over legal as last resort
    return {a: 1.0 / len(legal) for a in legal}

def _br_value(s, target: int, blueprint: Dict, value_fn, cfr_iters: int, depth_limit: int, depth_cap: int, memo: Dict) -> float:
    key = (public_state_key(s), tuple(tuple(p["hole"]) for p in s["players"]), s["to_act"], target, depth_cap)
    if key in memo:
        return memo[key]
    if is_chance_node(s):
        v = _br_value(chance_step(s), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap, memo)
        memo[key] = v
        return v
    if is_terminal(s):
        v = terminal_value(s, target)
        memo[key] = v
        return v
    if depth_cap <= 0:
        v = rollout_value(s, target, value_fn=value_fn)
        memo[key] = v
        return v
    player = s["to_act"]
    legal = legal_actions(s)
    if not legal:
        v = terminal_value(s, target)
        memo[key] = v
        return v
    if player == target:
        best = -1e18
        for a in legal:
            v = _br_value(step(s, a), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap - 1, memo)
            if v > best:
                best = v
        memo[key] = best
        return best
    else:
        sigma = _sigma_from_blueprint_or_resolve(s, blueprint, value_fn, cfr_iters, depth_limit)
        exp = 0.0
        for a, p in sigma.items():
            if p <= 0:
                continue
            exp += p * _br_value(step(s, a), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap - 1, memo)
        memo[key] = exp
        return exp

def _sigma_value(s, target: int, blueprint: Dict, value_fn, cfr_iters: int, depth_limit: int, depth_cap: int, memo: Dict) -> float:
    key = (public_state_key(s), tuple(tuple(p["hole"]) for p in s["players"]), s["to_act"], target, depth_cap, 'sigma')
    if key in memo:
        return memo[key]
    if is_chance_node(s):
        v = _sigma_value(chance_step(s), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap, memo)
        memo[key] = v
        return v
    if is_terminal(s):
        v = terminal_value(s, target)
        memo[key] = v
        return v
    if depth_cap <= 0:
        v = rollout_value(s, target, value_fn=value_fn)
        memo[key] = v
        return v
    sigma = _sigma_from_blueprint_or_resolve(s, blueprint, value_fn, cfr_iters, depth_limit)
    exp = 0.0
    for a, p in sigma.items():
        if p <= 0:
            continue
        exp += p * _sigma_value(step(s, a), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap - 1, memo)
    memo[key] = exp
    return exp

def _sample_state(num_players: int) -> dict:
    s = new_hand(btn=random.randrange(num_players), num_players=num_players)
    # random prefix to diversify
    for _ in range(random.randint(0, 6)):
        if is_chance_node(s):
            s = chance_step(s)
        else:
            la = legal_actions(s)
            if not la:
                break
            s = step(s, random.choice(la))
    return s

def cmd_exploitability_strict(args):
    bp = load_blueprint(args.blueprint)
    # value function (optional)
    if args.value and os.path.exists(args.value):
        value_fn = make_value_fn_from_dp(args.value) if args.value.endswith('.pt') else make_value_fn(load_value_params(args.value))
    else:
        value_fn = (lambda s, p: 0.0)
    samples = int(args.samples)
    num_players = int(args.num_players)
    depth_cap = int(args.depth_cap)
    cfr_iters = max(RESOLVE_ITERS, int(args.resolve_iters))
    depth_limit = max(RESOLVE_DEPTH, int(args.resolve_depth))
    total = 0.0
    used = 0
    for _ in trange(samples, desc="BR exploitability"):
        s = _sample_state(num_players)
        player = random.randrange(num_players)
        memo = {}
        try:
            br = _br_value(s, player, bp, value_fn, cfr_iters, depth_limit, depth_cap, memo)
            sv = _sigma_value(s, player, bp, value_fn, cfr_iters, depth_limit, depth_cap, memo)
            total += max(0.0, br - sv)
            used += 1
        except Exception:
            continue
    if used == 0:
        print(json.dumps({"error": "no_valid_samples"}))
        return
    avg = total / used
    mbb = (avg / BIG_BLIND) * 1000.0
    print(json.dumps({
        "samples": used,
        "avg_exploitability": avg,
        "mbb_per_hand": mbb,
        "resolve_iters": cfr_iters,
        "resolve_depth": depth_limit,
        "depth_cap": depth_cap
    }, indent=2))

def parse_action_str(a_str):
    a_str = a_str.strip().lower()
    if a_str in ['f','fold']: return ('F',)
    if a_str in ['x','check']: return ('X',)
    if a_str in ['c','call']: return ('C',)
    if a_str.startswith('r'):
        num=''.join(ch for ch in a_str if ch.isdigit())
        if num=='': raise ValueError("Use 'r <amount>' e.g., r 600")
        return ('R', int(num))
    raise ValueError("Unknown action. Use: f/x/c or r <amount>")

def cmd_play_cli(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    blueprint = load_blueprint(args.blueprint)
    value_fn = None
    if args.value:
        params = load_value_params(args.value)
        value_fn = make_value_fn(params)
    else:
        value_fn = lambda s,p: 0.0
    btn=0
    s = new_hand(btn=btn, num_players=args.num_players)
    print(f"[bold magenta]Hydra6 single-file CLI[/bold magenta]  (you are seat {args.human_seat})")
    print(f"Bot sits at seat {args.bot_seat}.")
    human=args.human_seat; bot=args.bot_seat
    while True:
        while is_chance_node(s): s = chance_step(s)
        if is_terminal(s):
            print("[bold]Hand finished[/bold]. Stacks:")
            for i,p in enumerate(s["players"]): print(f"Seat {i}: stack={p['stack']}")
            btn = (btn + 1) % s["num_players"]
            s = new_hand(btn=btn, num_players=args.num_players); continue
        cp = s["to_act"]
        if cp == human:
            print(f"[cyan]Stage:[/cyan] {s['stage_idx']}  [cyan]Board:[/cyan] {s['board']}")
            print(f"[cyan]Pot:[/cyan] {pot_size(s)}  [cyan]Current bet:[/cyan] {s['current_bet']}")
            me = s["players"][human]
            print(f"[green]Your hole:[/green] {me['hole']}  [green]Stack:[/green] {me['stack']}  [green]Invested:[/green] {me['invested']}")
            la = legal_actions(s); print(f"Legal: {la}")
            try:
                a = parse_action_str(Prompt.ask("Your action (f/x/c or r <amt>)"))
            except Exception as e:
                print(f"[red]{e}[/red]"); continue
            s = step(s, a)
        elif cp == bot:
            a = choose_bot_action(s, blueprint, value_fn, resolve_iters=args.resolve_iters, resolve_depth=args.resolve_depth)
            print(f"[yellow]Bot action:[/yellow] {a}")
            s = step(s, a)
        else:
            la = legal_actions(s); a=None
            if la:
                raises=[x for x in la if x[0]=='R']
                if raises and random.random()<0.1: a=random.choice(raises)
                else: a=('C',) if ('C',) in la else ('X',) if ('X',) in la else random.choice(la)
            if a is not None: s = step(s,a)

def state_from_json(d:dict):
    s = make_state(num_players=d["num_players"], sb=d.get("sb",50), bb=d.get("bb",100), ante=d.get("ante",0), btn=d.get("btn",0))
    s["players"] = [{
        "stack": int(p.get("stack", STARTING_STACK)),
        "invested": int(p.get("invested", 0)),
        "total_invested": int(p.get("total_invested", 0)),
        "folded": bool(p.get("folded", False)),
        "all_in": bool(p.get("all_in", False)),
        "hole": list(p.get("hole", []))
    } for p in d["players"]]
    s["stage_idx"] = {"preflop":0,"flop":1,"turn":2,"river":3}[d["stage"]]
    s["board"] = list(d.get("board", []))
    s["to_act"] = int(d["to_act"])
    s["current_bet"] = int(d.get("current_bet", 0))
    s["min_raise"] = int(d.get("min_raise", s["bb"]))
    s["pot"] = int(d.get("pot", 0))
    return s

def cmd_resolve_from_json(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    with open(args.situation,"r") as f: d=json.load(f)
    s = state_from_json(d)
    value_fn=None
    if args.value:
        params = load_value_params(args.value)
        value_fn = make_value_fn(params)
    else:
        value_fn = lambda st,pl: 0.0
    strat = depth_limited_resolve(s, value_fn=value_fn, cfr_iters=max(args.cfr_iters, RESOLVE_ITERS), depth_limit=max(args.depth, RESOLVE_DEPTH))
    if args.blueprint and not strat:
        bp = load_blueprint(args.blueprint)
        infoset = f"{public_state_key(s)}|priv:{private_obs(s, s['to_act'])}|p:{s['to_act']}"
        strat = bp.get(infoset, {})
    out = {str(k): float(v) for k,v in strat.items()}
    print(json.dumps(out, indent=2))

# -------- CLI --------
def main():
    p = argparse.ArgumentParser(description="Hydra6 single-file")
    sp = p.add_subparsers(dest="cmd", required=True)

    # selfplay
    a = sp.add_parser("selfplay_generate")
    a.add_argument("--episodes", type=int, default=2000)
    a.add_argument("--out", type=str, default="data/selfplay.npz")
    a.add_argument("--seed", type=int, default=123)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.set_defaults(func=cmd_selfplay_generate)

    # train_value
    a = sp.add_parser("train_value")
    a.add_argument("--data", type=str, required=True)
    a.add_argument("--save", type=str, default="value_single.pt")
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--lr", type=float, default=3e-4)
    a.set_defaults(func=cmd_train_value)

    # train_value_dist (multi-GPU)
    def _cmd_train_value_dist(args):
        return train_value_distributed(args.data, save_path=args.save, epochs=args.epochs, batch=args.batch, lr=args.lr)
    a = sp.add_parser("train_value_dist")
    a.add_argument("--data", type=str, required=True)
    a.add_argument("--save", type=str, default="value_dp.pt")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--batch", type=int, default=8192)
    a.add_argument("--lr", type=float, default=1e-3)
    a.set_defaults(func=_cmd_train_value_dist)

    # train_blueprint
    a = sp.add_parser("train_blueprint")
    a.add_argument("--iterations", type=int, default=20000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--save", type=str, default="blueprint.json")
    a.add_argument("--save-sum", type=str, default=None)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--use-abstraction", action="store_true")
    a.add_argument("--dynamic-abstraction", action="store_true")
    a.add_argument("--regret-decay", type=float, default=0.0)
    a.set_defaults(func=cmd_train_blueprint)

    # train_blueprint_dist (multi-process)
    a = sp.add_parser("train_blueprint_dist")
    a.add_argument("--iterations", type=int, default=100000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 1))
    a.add_argument("--save", type=str, default="blueprint_dist.json")
    a.add_argument("--save-sum", type=str, default=None)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--abstraction", type=str, default=None)
    a.add_argument("--use-abstraction", action="store_true")
    a.add_argument("--dynamic-abstraction", action="store_true")
    a.add_argument("--chunk", type=int, default=None)
    a.add_argument("--checkpoint-dir", type=str, default=None)
    # Auto-monitoring options
    a.add_argument("--eval-every-chunk", action="store_true")
    a.add_argument("--eval-hands", type=int, default=1000)
    a.add_argument("--eval-num-players", type=int, default=6)
    a.add_argument("--eval-baseline", type=str, choices=["random","tight"], default="random")
    a.add_argument("--eval-value", type=str, default=None)
    a.add_argument("--eval-abstraction", type=str, default=None)
    # Periodic head-to-head options
    a.add_argument("--h2h-every-chunk", action="store_true")
    a.add_argument("--h2h-blueprint", type=str, default=None)
    a.add_argument("--h2h-hands", type=int, default=1000)
    a.add_argument("--h2h-resolve-iters", type=int, default=None)
    a.add_argument("--h2h-resolve-depth", type=int, default=None)
    a.set_defaults(func=cmd_train_blueprint_dist)

    # merge
    a = sp.add_parser("merge_strategy_sums")
    a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--out-blueprint", type=str, default="blueprint_merged.json")
    a.add_argument("--out-sum", type=str, default=None)
    a.set_defaults(func=cmd_merge_sums)

    # play_cli
    a = sp.add_parser("play_cli")
    a.add_argument("--blueprint", type=str, required=True)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--bot-seat", type=int, default=2)
    a.add_argument("--human-seat", type=int, default=0)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--resolve-iters", type=int, default=None)
    a.add_argument("--resolve-depth", type=int, default=None)
    a.set_defaults(func=cmd_play_cli)

    # evaluate
    a = sp.add_parser("evaluate")
    a.add_argument("--blueprint", type=str, required=True)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--abstraction", type=str, default=None)
    a.add_argument("--baseline", type=str, choices=["random","tight"], default="random")
    a.add_argument("--hands", type=int, default=5000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--resolve-iters", type=int, default=None)
    a.add_argument("--resolve-depth", type=int, default=None)
    a.add_argument("--use-abstraction", action="store_true")
    a.add_argument("--dynamic-abstraction", action="store_true")
    a.set_defaults(func=cmd_evaluate)

    # strict exploitability (best response)
    a = sp.add_parser("exploitability_strict")
    a.add_argument("--blueprint", type=str, required=True)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--samples", type=int, default=100000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--depth-cap", type=int, default=4)
    a.add_argument("--resolve-iters", type=int, default=2000)
    a.add_argument("--resolve-depth", type=int, default=5)
    a.set_defaults(func=cmd_exploitability_strict)

    # smoke_test
    a = sp.add_parser("smoke_test")
    a.set_defaults(func=cmd_smoke_test)

    # quickcheck: 1-iteration train + tiny eval
    a = sp.add_parser("quickcheck")
    a.add_argument("--save", type=str, default="/Users/alexoht/Desktop/untitled folder/blueprint_quick.json")
    a.add_argument("--hands", type=int, default=4)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.set_defaults(func=cmd_quickcheck)

    # engine_sanity: validate core poker rules
    a = sp.add_parser("engine_sanity")
    a.set_defaults(func=cmd_engine_sanity)

    # resolve_from_json
    a = sp.add_parser("resolve_from_json")
    a.add_argument("--situation", type=str, required=True)
    a.add_argument("--blueprint", type=str, default=None)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--cfr-iters", type=int, default=200)
    a.add_argument("--depth", type=int, default=3)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.set_defaults(func=cmd_resolve_from_json)

    # abstraction build/load
    a = sp.add_parser("build_abstraction")
    a.add_argument("--data", type=str, required=True)
    a.add_argument("--hole-k", type=int, default=256)
    a.add_argument("--board-k", type=int, default=128)
    a.add_argument("--iters", type=int, default=25)
    a.add_argument("--save", type=str, default="abs.npz")
    a.add_argument("--potential-aware", action="store_true")
    def _cmd_build_abs(args):
        global PA_ABS
        PA_ABS = bool(args.potential_aware)
        return build_abstraction_from_dataset(args.data, args.hole_k, args.board_k, args.iters, args.save)
    a.set_defaults(func=_cmd_build_abs)

    a = sp.add_parser("load_abstraction")
    a.add_argument("--path", type=str, required=True)
    a.set_defaults(func=lambda args: load_abstraction(args.path))

    args = p.parse_args()
    random.seed(42)
    if hasattr(args, "seed"): random.seed(args.seed)
    args.func(args)

if __name__ == "__main__":
    main()
