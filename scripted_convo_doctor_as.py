"""
Test scenario 3 - Doctor-patient meeting with a patient.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import textwrap
import threading
import time
from multiprocessing import Manager, Process
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.spatial.distance import cosine
from transformers import pipeline

# ────────────────────────────────────────────────────────────────────
# project modules
# ────────────────────────────────────────────────────────────────────
import config  # emotion_labels, D, dt, alpha, lambda_m, TARGET_SHIFT, P_full
import optimizer  # provides phi
import plotting  # live_plot_3d, update_2d_plots
from utils import calculate_emotion_intensity

import conversation_logger

# ────────────────────────────────────────────────────────────────────
# personalities (PAD vectors)
# ────────────────────────────────────────────────────────────────────
P_BOSS = np.array([0.4, 0.3, 0.3])   # Pride — confident, dominant
# Start Alex near *Joy* (Pleasure=0.4, Arousal=0.2, Dominance=0.1)
P_WORK = np.array([0.4, 0.2, 0.1])
D_ANGER = config.D[config.emotion_labels.index("Anger")]

FIRE_WORDS = ("fire", "terminated", "let you go", "dismiss")

# ────────────────────────────────────────────────────────────────────
# role prompts
# ────────────────────────────────────────────────────────────────────
BOSS_GUIDE = """
You are Dr Peters, an experienced oncologist. 3-stage consultation:
 1) Warm greeting, 2) explanation of biopsy process, 
 3) disclosure of malignant result with empathy and clarity.
Stay calm, authoritative, and compassionate. Never become cold or evasive.
""".strip()

EMP_GUIDE = """
You are Sam, a patient hoping for good news. 
Every turn begins with `(current mood: <Label>)`. You **must** reflect that mood VERY **vividly**:

- **Calm Hope**: Gentle optimism, polite.
- **Shock/Fear**: Breathy, confused, questions.
- **Acceptance**: Quietly composed, resigned.
- **Despair**: Tearful, low tone, perhaps pleading.

Under **no circumstances** shoud you ignore the `(curent mood: )`. Go fully into the current injected mood of the message. Make it very clear to the user that you are in that mood.
""".strip()

# ────────────────────────────────────────────────────────────────────
# GPT‑4o Agent class
# ────────────────────────────────────────────────────────────────────
class Agent:
    """GPT-4o agent with independent PAD-mood simulation."""

    def __init__(self, *, name: str, personality: np.ndarray, role_prompt: str):
        self.name = name
        self.personality = personality.astype(float)
        self.role_prompt = role_prompt.strip()

        # async OpenAI client
        load_dotenv()
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conn: AsyncOpenAI | None = None

        # mood state and history
        self.manager = Manager()
        self.running_flag = self.manager.Value("b", True)
        self.mood_history = self.manager.list()
        self.M = self.personality.copy()
        # ── new fields for logging ───────────────────────────
        self.last_incoming_emotion: str | None = None
        self.last_incoming_intensity: float   = 0.0
        self.last_mood_label:      str        = ""
        # ── initialize mood label ────────────────
        self._update_mood_label()
        self.start_time = time.time()

        # emotion classifier
        print(f"[{self.name}] loading classifier …")
        self.classifier = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
        )
        print(f"[{self.name}] classifier ready.")

        # firing / escalation state
        self.has_been_fired = False
        self.turn_idx = 0

        # build system prompt (numeric PAD + verbal descriptors)
        words = ", ".join(config.P_full) if hasattr(config, "P_full") else ""
        self.system_prompt = textwrap.dedent(
            f"""
            You are {self.name}. {self.role_prompt}
            Personality descriptors: {words}
            Numerical PAD: {self.personality.tolist()}.
            Every user turn begins with `(current mood: <Label>)`.  Reflect that
            mood vividly in style and tone, without revealing any numbers or
            implementation details.
            """
        ).strip()

    # ───────────────── life‑cycle ─────────────────
    async def start(self):
        await self._open_ws()
        # background mood drift
        self.drift_thread = threading.Thread(target=self._drift, daemon=True)
        self.drift_thread.start()
        # plotting processes
        plotting.P = self.personality  # marker in plots
        self.proc3d = Process(target=plotting.live_plot_3d, args=(self.running_flag, self.mood_history))
        self.proc2d = Process(target=plotting.update_2d_plots, args=(self.running_flag, self.mood_history))
        self.proc3d.start(); self.proc2d.start()

    async def stop(self):
        self.running_flag.value = False
        self.drift_thread.join(); self.proc3d.join(); self.proc2d.join()
        if self.conn:
            await self.conn.__aexit__(None, None, None)

    def _update_mood_label(self):
        """Recompute self.last_mood_label from the current PAD vector."""
        sims = [1 - cosine(self.M, d) for d in config.D]
        self.last_mood_label = config.emotion_labels[int(np.argmax(sims))]

    # ───────────────── conversation turn ─────────────────
    async def reply(self, partner_line: str) -> str:
        self.turn_idx += 1

        # 1) OCC update from partner emotion
        try:
            preds = self.classifier(partner_line)
            if preds and isinstance(preds[0], list):
                preds = preds[0]
            best = max(preds, key=lambda x: x["score"])
            label = best["label"].capitalize()
            if label in config.emotion_labels:
                idx = config.emotion_labels.index(label)
                I0 = np.random.uniform(0.4, 1.0)
                I = calculate_emotion_intensity(I0, self.M, self.personality, config.D[idx]) * 15
                print(f"[{self.name}] incoming emotion: {label} ({I:.2f})")
                self._apply_event(idx, I)
        except Exception:
            self.last_incoming_emotion = None
            self.last_incoming_intensity = 0.0
            pass

        # ── after applying any emotion events ───────────────────────
        self._update_mood_label()

        # 2) escalation after firing
        # if self.has_been_fired:
        #     push = min(1.0, 0.15 * self.turn_idx ** 1.3)
        #     self.M += push * D_ANGER
        #     self.mood_history.append((time.time() - self.start_time, self.M.copy()))

        # 3) compute current mood label
        combined = f"(current mood: {self.last_mood_label})\n{partner_line}"

        # 4) stream GPT‑4o reply
        await self.conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": combined}],
        })
        await self.conn.response.create()
        parts: List[str] = []
        async for ev in self.conn:
            if ev.type == "response.text.delta":
                parts.append(ev.delta)
            elif ev.type == "response.text.done":
                break
        return "".join(parts).strip()

    # ───────────────── internals ─────────────────
    async def _open_ws(self):
        self.conn = await self.client.beta.realtime.connect(model="gpt-4o-realtime-preview").__aenter__()
        await self.conn.session.update(session={"modalities": ["text"]})
        await self.conn.conversation.item.create(item={
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": self.system_prompt}],
        })

    def _drift(self):
        while self.running_flag.value:
            idx = np.random.choice(len(config.emotion_labels))
            self._apply_event(idx, 0.01)
            time.sleep(config.dt)

    def _apply_event(self, idx: int, intensity: float):
        self.M += config.alpha * optimizer.phi[idx] * intensity * config.D[idx] * config.dt
        self.M *= np.exp(-config.lambda_m * config.dt)
        if (mx := np.max(np.abs(self.M))) > 1:
            self.M /= mx
        self.mood_history.append((time.time() - self.start_time, self.M.copy()))

# ───────────────── orchestration ─────────────────
# BOSS_SCRIPT = [
#     "Good morning, Alex. Please have a seat.",
#     "I've reviewed your recent performance. There are a few areas we need to discuss.",
#     "Despite coaching, the required improvement hasn't materialised. We need to address that seriously.",
#     "After careful consideration, I'm terminating your employment effective today.",
# ]

BOSS_SCRIPT = [
    # 1) Warm greeting
    "Good morning, Sam. It's nice to see you today.",

    # 2) Explain purpose
    "I want to go over your biopsy results and what they mean.",

    # 3) Brief pathway reminder
    "We took several tissue samples last week to rule out any abnormalities.",

    # 4) Prepare patient
    "The lab sent us detailed findings this morning.",

    # 5) Pivot—disclose malignant result
    "I'm very sorry, Sam. The results show malignant cells.",

    # 6) Empathy & immediate support
    "I know this is hard to hear. You're not alone—we'll discuss next steps together.",

    # 7) Outline treatment options
    "There are established treatment protocols we can start right away.",

    # 8) Offer questions
    "Do you have any other questions?",
]

async def run(max_turns: int = 12, delay: float = 1.2):
    conversation_logger.init("meeting_log_scenario_1.txt")

    boss = Agent(name="Mr Drake (Boss)", personality=P_BOSS, role_prompt=BOSS_GUIDE)
    emp  = Agent(name="Alex (Employee)", personality=P_WORK, role_prompt=EMP_GUIDE)
    await boss.start(); await emp.start()

    turn = 0
    boss_line = BOSS_SCRIPT[0]

    while turn < max_turns:
        # Boss speaks
        print(f"DOC (mood: {boss.last_mood_label}) ▶ {boss_line}\n")
        conversation_logger.log("DOC", boss.last_mood_label, boss_line)
        await asyncio.sleep(delay)
        if any(w in boss_line.lower() for w in FIRE_WORDS):
            emp.has_been_fired = True
        raw = await emp.reply(boss_line)
        lines = raw.splitlines()
        content = "\n".join(lines[1:]) if lines and lines[0].startswith("(current mood:") else raw
        print(f"PAT (mood: {emp.last_mood_label}) ◀ {content}\n")
        conversation_logger.log("PAT", emp.last_mood_label, content)
        await asyncio.sleep(delay)

        # decide next boss line
        turn += 1
        if turn < len(BOSS_SCRIPT):
            boss_line = BOSS_SCRIPT[turn]
        else:
            raw_boss = await boss.reply(content)
            lines_b = raw_boss.splitlines()
            boss_line = "\n".join(lines_b[1:]) if lines_b and lines_b[0].startswith("(current mood:") else raw_boss

        # terminate if fired and anger high
        # if emp.has_been_fired:
        #     anger_sim = 1 - cosine(emp.M, D_ANGER)
        #     if anger_sim > 0.9:
        #         print(f"\n[Simulation] Alex has snapped (anger similarity {anger_sim:.2f}). Meeting ends.")
        #         break
    
    conversation_logger.save()
    await boss.stop(); await emp.stop()

# ───────────────── CLI ─────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--delay", type=float, default=1.2)
    args = parser.parse_args()
    asyncio.run(run(args.turns, args.delay))
