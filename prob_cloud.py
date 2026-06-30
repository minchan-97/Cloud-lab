"""
prob_cloud.py — 확률 구름 표상 (가스코드 철학의 구현)

발상 (민찬기): 전자구름처럼 —
  "존재하지만 어디 있는지 모른다. 관측하면 정해진다."

구현:
  - 입력은 한 노드(BMU, 입자)가 아니라, SOM 노드들 위의 '확률 구름'으로 표상.
    psi_i = exp(-거리_i / temp) 를 정규화 → 확률밀도 |psi|^2 흉내.
  - temp(온도)가 높을수록 구름이 넓게 퍼짐(불확정), 낮을수록 한 점(입자).
  - '관측'(질문)하면 구름이 가장 확률 높은 노드로 '붕괴'(collapse).

이건 진짜 양자역학(복소수/위상/슈뢰딩거)이 아니라,
'확률장 + 관측 붕괴'라는 구조만 빌린 고전적 흉내다. 정직하게.
"""
from __future__ import annotations
import numpy as np


def cloud_distribution(W, x, temp=1.0):
    """
    입력 x 에 대한 SOM 노드들 위의 확률 구름.
    psi_i ∝ exp(-dist_i / temp), 정규화하여 확률(|psi|^2 흉내) 반환.
    temp ↑ → 넓게 퍼짐(불확정), temp ↓ → 한 점(확정).
    """
    d = np.linalg.norm(W - x, axis=1)
    # 거리를 음의 로그확률처럼 사용 → 소프트맥스로 확률화
    z = -d / max(temp, 1e-6)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum() + 1e-12
    return p


def cloud_entropy(p):
    """구름의 퍼짐 정도(섀넌 엔트로피). 높을수록 넓게 퍼짐(불확정)."""
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def max_entropy(n):
    """노드 n개일 때 최대 엔트로피(완전 균등 분포)."""
    return float(np.log(n))


def observe_collapse(p):
    """
    관측: 구름이 가장 확률 높은 노드로 붕괴.
    반환: (붕괴된 노드 idx, 그 노드의 확률=붕괴 확신도)
    """
    idx = int(np.argmax(p))
    return idx, float(p[idx])


def participation_ratio(p):
    """
    '몇 개의 노드가 실질적으로 구름에 참여하나' (역-단순도).
    1 = 완전히 한 노드(입자), n = 완전히 퍼짐(구름).
    """
    return float(1.0 / np.sum(p ** 2))


# ======================================================================
# 관측 붕괴의 세 측면: 속도 / 용량 / 다양성
# ======================================================================
def collapse_speed(W, x, temps=None):
    """
    붕괴 속도: temp를 낮춰갈 때 '확신 0.9 이상'에 도달하는 시점.
    낮은 temp에서 빨리 확정될수록 붕괴가 '빠름'.
    반환: (확신이 0.9 넘는 첫 temp, 그때 참여노드수) — 못 넘으면 (None, ...)
    """
    if temps is None:
        temps = [5.0, 3.0, 2.0, 1.0, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    d = np.linalg.norm(W - x, axis=1)
    traj = []
    collapse_temp = None
    for t in temps:
        z = -d / max(t, 1e-6); z -= z.max()
        p = np.exp(z); p /= p.sum() + 1e-12
        conf = float(p.max()); pr = float(1.0/np.sum(p**2))
        traj.append((t, conf, pr))
        if collapse_temp is None and conf >= 0.9:
            collapse_temp = t
    return collapse_temp, traj


def cloud_capacity(W, x, temp=1.0, energy=0.95):
    """
    용량: 구름이 '실질적으로' 담는 상태(노드) 수.
    - participation_ratio: 유효 노드 수
    - effective_states: 누적확률 energy(95%)를 채우는 데 필요한 노드 수
    """
    d = np.linalg.norm(W - x, axis=1)
    z = -d / max(temp, 1e-6); z -= z.max()
    p = np.exp(z); p /= p.sum() + 1e-12
    pr = float(1.0/np.sum(p**2))
    sp = np.sort(p)[::-1]
    cum = np.cumsum(sp)
    eff = int(np.searchsorted(cum, energy) + 1)
    return {"participation": pr, "effective_states": eff, "total_nodes": len(p)}


def collapse_diversity(W, inputs, temp=0.1):
    """
    다양성: 여러 입력을 관측 붕괴시켰을 때, 서로 다른 노드로 붕괴하는 정도.
    - unique_nodes: 붕괴 결과의 고유 노드 수
    - coverage: 전체 노드 중 몇 %가 붕괴 목적지로 쓰였나
    - distribution_entropy: 붕괴 목적지 분포의 엔트로피(고를수록 다양)
    """
    landing = []
    for x in inputs:
        d = np.linalg.norm(W - x, axis=1)
        landing.append(int(np.argmin(d)))
    landing = np.array(landing)
    uniq = len(set(landing.tolist()))
    counts = np.bincount(landing, minlength=len(W)).astype(float)
    pc = counts[counts > 0] / counts.sum()
    ent = float(-np.sum(pc * np.log(pc)))
    return {"unique_nodes": uniq, "total_nodes": len(W),
            "coverage": round(uniq/len(W), 3),
            "landing_entropy": round(ent, 3),
            "max_entropy": round(float(np.log(len(W))), 3)}


# ======================================================================
# 확률 구름 레이어 쌓기 (역전파 없음 — 각 층은 독립 SOM, 확률만 흐름)
# ======================================================================
class CloudLayer:
    """한 층: 입력 벡터 → SOM 노드들 위의 확률 구름 → (선택)다음 층 입력 벡터."""
    def __init__(self, W, temp=1.0):
        self.W = W           # 이 층의 SOM 노드들
        self.temp = temp

    def forward(self, x, collapse=False):
        """
        x(벡터) → 확률 구름 p.
        collapse=False: 구름 유지 → 다음 층 입력은 '확률가중 평균 벡터'(구름의 중심)
        collapse=True : 붕괴 → 다음 층 입력은 '가장 확률 높은 노드 벡터'(한 점)
        반환: (p 확률분포, 다음층_입력벡터)
        """
        d = np.linalg.norm(self.W - x, axis=1)
        z = -d / max(self.temp, 1e-6); z -= z.max()
        p = np.exp(z); p /= p.sum() + 1e-12
        if collapse:
            nxt = self.W[int(np.argmax(p))]
        else:
            nxt = (p[:, None] * self.W).sum(axis=0)  # 확률가중 평균 = 구름 중심
        return p, nxt


def stack_forward(layers, x, collapse_between=False):
    """
    여러 CloudLayer를 통과시키며 각 층의 구름을 기록.
    collapse_between: 층 사이에서 붕괴시킬지(True=점 전달, False=구름중심 전달)
    반환: 층별 확률분포 리스트
    """
    clouds = []
    cur = x
    for L in layers:
        p, cur = L.forward(cur, collapse=collapse_between)
        clouds.append(p)
    return clouds


# ======================================================================
# 영역별 구름 (가로 병렬) — 각 영역이 자기 SOM 위에 구름을 형성하고
# 관측 시 종합한다. 깊이(세로)가 아니라 관점(가로)으로 쌓는 구조.
# ======================================================================
class DomainCloud:
    """한 영역(도메인/정체성/이상 등)의 구름. 자기 SOM + 자기 기준 거리."""
    def __init__(self, name, temp=1.0):
        self.name = name
        self.temp = temp
        self.W = None           # 이 영역의 SOM 노드
        self.ref_dist = None    # 정상 입력의 평균 BMU 거리(이상 판정 기준)

    def fit(self, X, n_nodes=20, rounds=30, seed=0):
        """영역 데이터 X로 SOM을 자라게 하고, 정상 거리 기준을 잡는다."""
        import numpy as np
        from selfloop_engine import GrowingSOM
        n = min(n_nodes, len(X))
        g = GrowingSOM(dim=X.shape[1], init_nodes=n, seed=seed)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n, replace=False)
        g.W = X[idx].copy() + rng.normal(scale=0.02, size=(n, X.shape[1]))
        g.coords = g.coords[:n]; g.err = g.err[:n]
        for _ in range(rounds):
            g.round += 1; g.train_step(X, 0.1, 1.0)
        self.W = g.W
        # 정상 입력들의 BMU 거리 분포 → 이상 판정 기준
        bmu_d = np.min(np.linalg.norm(X[:, None, :] - self.W[None, :, :], axis=2), axis=1)
        self.ref_dist = (float(np.mean(bmu_d)), float(np.std(bmu_d)))
        return self

    def cloud(self, x):
        """입력 x의 이 영역 구름 + 정상 여부 신호."""
        import numpy as np
        d = np.linalg.norm(self.W - x, axis=1)
        z = -d / max(self.temp, 1e-6); z -= z.max()
        p = np.exp(z); p /= p.sum() + 1e-12
        bmu = float(np.min(d))
        mean, std = self.ref_dist
        # 정상 거리 기준에서 몇 표준편차 벗어났나 (z-score, 클수록 이상)
        deviation = (bmu - mean) / (std + 1e-9)
        return {"p": p, "bmu_dist": bmu, "deviation": float(deviation),
                "entropy": float(-np.sum(p[p>1e-12]*np.log(p[p>1e-12])))}


class CloudEnsemble:
    """여러 영역 구름의 종합 — 관측 시 각 영역 의견을 모아 판정+설명."""
    def __init__(self, dev_threshold=2.0):
        self.domains = {}
        self.dev_threshold = dev_threshold  # 이 표준편차 넘으면 그 영역이 '이상'

    def add(self, domain_cloud):
        self.domains[domain_cloud.name] = domain_cloud
        return self

    def observe(self, x):
        """관측: 모든 영역 구름을 보고 종합 판정 + 어느 영역이 문제인지."""
        report = {}
        flagged = []
        for name, dc in self.domains.items():
            c = dc.cloud(x)
            report[name] = {"deviation": round(c["deviation"], 2),
                            "entropy": round(c["entropy"], 2)}
            if c["deviation"] > self.dev_threshold:
                flagged.append(name)
        verdict = "정상" if not flagged else f"이상: {', '.join(flagged)} 영역 벗어남"
        return {"verdict": verdict, "flagged": flagged, "report": report}


# ======================================================================
# 구름 생성기 (생성형의 씨앗)
#   A: 구름(영역별 부드러운 데이터)에서 마르코프가 토큰을 이어붙여 생성
#   B: 생성된 후보들을 마르코프 품질(logP)로 선별
# "마르코프로 구름을 관측한다" = 구름의 가능성 + 마르코프의 품질
# ======================================================================
class CloudGenerator:
    def __init__(self, markov, cloud_vocab, temp=1.0):
        """
        markov: 학습된 MarkovGuardrail (품질 제어기)
        cloud_vocab: 이 영역 구름에 속한 단어 집합 (생성 재료 = 사용자 학습 결과)
        temp: 생성 다양성 (높을수록 모험적)
        """
        self.m = markov
        self.vocab = list(cloud_vocab)
        self.temp = temp

    def _next_token(self, prev, rng):
        """prev 다음에 올 토큰을, 구름어휘 중에서 마르코프 확률로 샘플링."""
        import numpy as np
        cands = self.vocab
        logps = np.array([self.m._logP(prev, w) for w in cands])
        z = logps / max(self.temp, 1e-6)
        z -= z.max()
        p = np.exp(z); p /= p.sum() + 1e-12
        return cands[int(rng.choice(len(cands), p=p))]

    def generate(self, start="<s>", max_len=12, seed=0):
        """A: 구름에서 한 문장 생성 (마르코프가 관측하며 이어붙임)."""
        import numpy as np
        rng = np.random.default_rng(seed)
        out = []
        prev = start
        for _ in range(max_len):
            w = self._next_token(prev, rng)
            if w in ("</s>", "<s>"):
                break
            out.append(w)
            prev = w
        return " ".join(out)

    def generate_and_select(self, n_candidates=10, max_len=12, seed=0):
        """
        A+B: 여러 후보 생성 후, 마르코프 품질(평균 logP)로 선별.
        반환: [(문장, 품질점수)] 품질 높은 순.
        """
        import numpy as np
        cands = []
        for i in range(n_candidates):
            s = self.generate(max_len=max_len, seed=seed + i)
            if not s.strip():
                continue
            score = self.m.score(s)  # 마르코프 품질 (높을수록 자연스러움)
            cands.append((s, round(score, 2)))
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands


# ======================================================================
# 생성 품질 개선 (화이트박스 유지 — 모든 결정이 셀 수 있는 확률)
#   1) 트라이그램: 두 단어 앞을 봐 맥락 연장
#   2) 구름 확률 가중: 영역 구름의 분포로 단어 선택 편향
#   3) 반복 억제: 방금 쓴 단어 회피
# ======================================================================
class TrigramCloudGenerator:
    def __init__(self, sentences, cloud_weights=None, temp=0.7,
                 repeat_penalty=3.0, jm=(0.6, 0.3, 0.1)):
        """
        sentences: 학습 문장 (트라이/바이/유니그램 카운트 구축)
        cloud_weights: {단어: 가중치} 영역 구름 분포 (없으면 균등)
        jm: (트라이, 바이, 유니) 보간 계수
        """
        import math
        from collections import Counter
        from selfloop_engine import tokenize
        self.tok = tokenize
        self.temp = temp
        self.repeat_penalty = repeat_penalty
        self.jm = jm
        self.uni = Counter(); self.bi = Counter(); self.tri = Counter()
        self.uni_total = 0
        self.vocab = set()
        for s in sentences:
            t = ["<s>", "<s>"] + tokenize(s) + ["</s>"]
            for i in range(len(t)):
                self.uni[t[i]] += 1; self.uni_total += 1; self.vocab.add(t[i])
                if i >= 1: self.bi[(t[i-1], t[i])] += 1
                if i >= 2: self.tri[(t[i-2], t[i-1], t[i])] += 1
        self.cloud_w = cloud_weights or {}
        self._math = math

    def _logP(self, w1, w2, w3):
        """트라이그램 JM 보간 확률 (두 단어 앞 w1,w2 → w3)."""
        V = max(1, len(self.vocab))
        p_uni = (self.uni.get(w3, 0) + 1) / (self.uni_total + V)
        c_bi = self.uni.get(w2, 0)
        p_bi = (self.bi.get((w2, w3), 0) / c_bi) if c_bi > 0 else 0.0
        c_tri = self.bi.get((w1, w2), 0)
        p_tri = (self.tri.get((w1, w2, w3), 0) / c_tri) if c_tri > 0 else 0.0
        a, b, c = self.jm
        p = a * p_tri + b * p_bi + c * p_uni
        return self._math.log(max(p, 1e-10))

    def generate(self, max_len=14, seed=0, candidate_pool=80):
        import numpy as np
        rng = np.random.default_rng(seed)
        cand_words = [w for w, _ in self.uni.most_common(candidate_pool)
                      if w not in ("<s>", "</s>")]
        out = []; w1, w2 = "<s>", "<s>"
        recent = {}
        for _ in range(max_len):
            logps = []
            for w in cand_words:
                lp = self._logP(w1, w2, w)
                # 구름 가중(영역성 강화)
                if self.cloud_w:
                    lp += np.log(self.cloud_w.get(w, 1e-3))
                # 반복 억제
                if w in recent:
                    lp -= self.repeat_penalty * recent[w]
                logps.append(lp)
            logps = np.array(logps)
            z = logps / max(self.temp, 1e-6); z -= z.max()
            p = np.exp(z); p /= p.sum() + 1e-12
            nxt = cand_words[int(rng.choice(len(cand_words), p=p))]
            if nxt == "</s>":
                break
            out.append(nxt)
            recent = {k: v * 0.5 for k, v in recent.items()}
            recent[nxt] = recent.get(nxt, 0) + 1
            w1, w2 = w2, nxt
        return " ".join(out)

    def score(self, sentence):
        """생성물 품질(평균 트라이그램 logP)."""
        import numpy as np
        t = ["<s>", "<s>"] + self.tok(sentence) + ["</s>"]
        lps = [self._logP(t[i-2], t[i-1], t[i]) for i in range(2, len(t))]
        return float(np.mean(lps)) if lps else -99.0

    def generate_and_select(self, n_candidates=12, max_len=14, seed=0):
        cands = []
        for i in range(n_candidates):
            s = self.generate(max_len=max_len, seed=seed + i)
            if s.strip():
                cands.append((s, round(self.score(s), 2)))
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands


# ======================================================================
# 가스코드 구름 × n-gram 연결 생성기
#   매 스텝: 구름(SOM 확률장)이 "가능한 단어 영역"을 주고,
#            n-gram이 "이어질 단어"를 주어, 둘을 곱해 다음 단어 결정.
#   구름과 마르코프가 서로 제약하며 체인이 이어진다.
# ======================================================================
class GasCloudNgram:
    def __init__(self, sentences, emb, som_W, temp_cloud=1.0, temp_gen=0.7,
                 jm=(0.6, 0.3, 0.1), repeat_penalty=2.0):
        import math
        from collections import Counter
        from selfloop_engine import tokenize
        self.tok = tokenize
        self.emb = emb
        self.W = som_W                  # 가스코드 구름의 SOM 노드들
        self.temp_cloud = temp_cloud
        self.temp_gen = temp_gen
        self.jm = jm
        self.repeat_penalty = repeat_penalty
        self._math = math
        # n-gram 카운트
        self.uni = Counter(); self.bi = Counter(); self.tri = Counter()
        self.uni_total = 0; self.vocab = set()
        self.word_vec = {}              # 단어 → 임베딩 (구름 투영용)
        for s in sentences:
            toks = tokenize(s)
            for w in toks:
                if w not in self.word_vec:
                    self.word_vec[w] = emb.encode(w)
            t = ["<s>", "<s>"] + toks + ["</s>"]
            for i in range(len(t)):
                self.uni[t[i]] += 1; self.uni_total += 1; self.vocab.add(t[i])
                if i >= 1: self.bi[(t[i-1], t[i])] += 1
                if i >= 2: self.tri[(t[i-2], t[i-1], t[i])] += 1

    def _ngram_logP(self, w1, w2, w3):
        V = max(1, len(self.vocab))
        p_uni = (self.uni.get(w3, 0) + 1) / (self.uni_total + V)
        c_bi = self.uni.get(w2, 0)
        p_bi = (self.bi.get((w2, w3), 0) / c_bi) if c_bi > 0 else 0.0
        c_tri = self.bi.get((w1, w2), 0)
        p_tri = (self.tri.get((w1, w2, w3), 0) / c_tri) if c_tri > 0 else 0.0
        a, b, c = self.jm
        return self._math.log(max(a*p_tri + b*p_bi + c*p_uni, 1e-10))

    def _cloud_logP(self, context_vec, w):
        """
        가스코드 구름 기본식: 단어 w가 현재 문맥의 '구름' 안에 있는 정도.
        context_vec(현재까지 문맥의 구름 중심)와 단어 임베딩의 거리를
        SOM 확률장으로 변환 → 구름 소속 로그확률.
        """
        import numpy as np
        wv = self.word_vec.get(w)
        if wv is None:
            return -10.0
        # 단어를 SOM 구름에 투영: 노드들과의 거리 → 확률장
        d_nodes = np.linalg.norm(self.W - wv, axis=1)
        z = -d_nodes / max(self.temp_cloud, 1e-6); z -= z.max()
        p_cloud = np.exp(z); p_cloud /= p_cloud.sum() + 1e-12
        # 문맥 구름 중심과의 정합도 (문맥과 같은 구름 영역인가)
        d_ctx = np.linalg.norm(context_vec - wv)
        cloud_fit = -d_ctx / max(self.temp_cloud, 1e-6)
        return float(cloud_fit)

    def generate(self, max_len=14, seed=0, candidate_pool=120, alpha=0.5):
        """
        alpha: 구름 vs n-gram 비중 (0=n-gram만, 1=구름만, 0.5=반반)
        매 스텝 두 확률장을 곱(로그합)해 다음 단어 결정 → 체인 이어짐.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        cand_words = [w for w, _ in self.uni.most_common(candidate_pool)
                      if w not in ("<s>", "</s>")]
        out = []; w1, w2 = "<s>", "<s>"
        recent = {}
        # 문맥 구름 중심 (생성되며 갱신)
        ctx_vec = np.zeros(self.W.shape[1])
        ctx_n = 0
        for _ in range(max_len):
            scores = []
            for w in cand_words:
                lp_ngram = self._ngram_logP(w1, w2, w)
                lp_cloud = self._cloud_logP(ctx_vec if ctx_n > 0 else self.word_vec.get(w, ctx_vec), w)
                # 구름 × n-gram (로그합 = 곱) — 둘이 서로 제약
                s = (1 - alpha) * lp_ngram + alpha * lp_cloud
                if w in recent:
                    s -= self.repeat_penalty * recent[w]
                scores.append(s)
            scores = np.array(scores)
            z = scores / max(self.temp_gen, 1e-6); z -= z.max()
            p = np.exp(z); p /= p.sum() + 1e-12
            nxt = cand_words[int(rng.choice(len(cand_words), p=p))]
            if nxt == "</s>":
                break
            out.append(nxt)
            # 문맥 구름 갱신 (생성된 단어로 구름 중심 이동)
            wv = self.word_vec.get(nxt)
            if wv is not None:
                ctx_vec = (ctx_vec * ctx_n + wv) / (ctx_n + 1); ctx_n += 1
            recent = {k: v*0.5 for k, v in recent.items()}
            recent[nxt] = recent.get(nxt, 0) + 1
            w1, w2 = w2, nxt
        return " ".join(out)

    def score(self, sentence):
        import numpy as np
        t = ["<s>", "<s>"] + self.tok(sentence) + ["</s>"]
        lps = [self._ngram_logP(t[i-2], t[i-1], t[i]) for i in range(2, len(t))]
        return float(np.mean(lps)) if lps else -99.0


# ======================================================================
# 프랙탈 3중 n-gram 생성기 (시작·중간·결말 트라이앵글 + 구름 접착)
#   - 문장을 위치별 3구간(시작/중간/결말)으로 나눠 각각 n-gram 학습
#   - 각 구간은 구름×n-gram으로 생성 (영역 유지 + 반복 억제)
#   - 구간 경계는 구름이 '접착제'로 매끄럽게 연결
#   1차원 단일체인의 문법천장을, 문장 뼈대 구조로 보완.
# ======================================================================
class FractalTripleNgram:
    def __init__(self, sentences, emb, som_W, temp_cloud=1.0, temp_gen=0.6,
                 jm=(0.6, 0.3, 0.1), repeat_penalty=2.0):
        import math
        from collections import Counter
        from selfloop_engine import tokenize
        self.tok = tokenize; self.emb = emb; self.W = som_W
        self.temp_cloud = temp_cloud; self.temp_gen = temp_gen
        self.jm = jm; self.repeat_penalty = repeat_penalty; self._math = math
        # 위치별 3구간 n-gram (시작/중간/결말)
        self.seg = {"start": self._empty(), "mid": self._empty(), "end": self._empty()}
        self.vocab = set(); self.word_vec = {}
        for s in sentences:
            toks = tokenize(s)
            if len(toks) < 3:
                continue
            for w in toks:
                if w not in self.word_vec:
                    self.word_vec[w] = emb.encode(w)
                self.vocab.add(w)
            # 문장을 3등분 → 각 구간에 해당 부분 학습
            L = len(toks); a, b = L // 3, 2 * L // 3
            parts = {"start": ["<s>", "<s>"] + toks[:a+1],
                     "mid":   [toks[a-1] if a > 0 else "<s>", toks[a] if a < L else "<s>"] + toks[a:b+1],
                     "end":   [toks[b-1] if b > 0 else "<s>", toks[b] if b < L else "<s>"] + toks[b:] + ["</s>"]}
            for name, t in parts.items():
                self._add(self.seg[name], t)

    def _empty(self):
        from collections import Counter
        return {"uni": Counter(), "bi": Counter(), "tri": Counter(), "total": 0}

    def _add(self, seg, t):
        for i in range(len(t)):
            seg["uni"][t[i]] += 1; seg["total"] += 1
            if i >= 1: seg["bi"][(t[i-1], t[i])] += 1
            if i >= 2: seg["tri"][(t[i-2], t[i-1], t[i])] += 1

    def _ngram_logP(self, seg, w1, w2, w3):
        V = max(1, len(self.vocab))
        p_uni = (seg["uni"].get(w3, 0) + 1) / (seg["total"] + V)
        c_bi = seg["uni"].get(w2, 0)
        p_bi = (seg["bi"].get((w2, w3), 0) / c_bi) if c_bi > 0 else 0.0
        c_tri = seg["bi"].get((w1, w2), 0)
        p_tri = (seg["tri"].get((w1, w2, w3), 0) / c_tri) if c_tri > 0 else 0.0
        a, b, c = self.jm
        return self._math.log(max(a*p_tri + b*p_bi + c*p_uni, 1e-10))

    def _cloud_glue(self, ctx_vec, w):
        """구름 접착제: 단어 w가 현재 문맥 구름과 얼마나 정합하나."""
        import numpy as np
        wv = self.word_vec.get(w)
        if wv is None:
            return -10.0
        return float(-np.linalg.norm(ctx_vec - wv) / max(self.temp_cloud, 1e-6))

    def _gen_segment(self, seg, ctx_vec, ctx_n, w1, w2, recent, rng,
                     cand_words, n_words, alpha):
        """한 구간 생성. 구름×n-gram. 끝 상태(ctx,w1,w2)를 반환해 다음 구간이 이어받음."""
        import numpy as np
        out = []
        for _ in range(n_words):
            scores = []
            for w in cand_words:
                lp = (1 - alpha) * self._ngram_logP(seg, w1, w2, w)
                if ctx_n > 0:
                    lp += alpha * self._cloud_glue(ctx_vec, w)
                if w in recent:
                    lp -= self.repeat_penalty * recent[w]
                scores.append(lp)
            scores = np.array(scores)
            z = scores / max(self.temp_gen, 1e-6); z -= z.max()
            p = np.exp(z); p /= p.sum() + 1e-12
            nxt = cand_words[int(rng.choice(len(cand_words), p=p))]
            if nxt == "</s>":
                break
            out.append(nxt)
            wv = self.word_vec.get(nxt)
            if wv is not None:
                ctx_vec = (ctx_vec * ctx_n + wv) / (ctx_n + 1); ctx_n += 1
            recent = {k: v*0.5 for k, v in recent.items()}
            recent[nxt] = recent.get(nxt, 0) + 1
            w1, w2 = w2, nxt
        return out, ctx_vec, ctx_n, w1, w2, recent

    def generate(self, total_len=15, seed=0, candidate_pool=120, alpha=0.5):
        """시작→중간→결말, 구름이 구간을 접착하며 이어붙임."""
        import numpy as np
        rng = np.random.default_rng(seed)
        cand_words = [w for w, _ in
                      (self.seg["start"]["uni"] + self.seg["mid"]["uni"] + self.seg["end"]["uni"]).most_common(candidate_pool)
                      if w not in ("<s>", "</s>")]
        ctx_vec = np.zeros(self.W.shape[1]); ctx_n = 0
        w1, w2 = "<s>", "<s>"; recent = {}
        parts = []
        seg_len = max(2, total_len // 3)
        for name in ["start", "mid", "end"]:
            words, ctx_vec, ctx_n, w1, w2, recent = self._gen_segment(
                self.seg[name], ctx_vec, ctx_n, w1, w2, recent, rng,
                cand_words, seg_len, alpha)
            parts += words
            # 구간 경계: 구름 문맥은 유지(접착), n-gram 상태도 이어받음
        return " ".join(parts)

    def score(self, sentence):
        """전체 품질 — 위치별 구간 n-gram 평균."""
        import numpy as np
        toks = self.tok(sentence)
        if len(toks) < 3:
            return -99.0
        L = len(toks); a, b = L//3, 2*L//3
        segs = [("start", ["<s>","<s>"]+toks[:a+1]),
                ("mid", ["<s>","<s>"]+toks[a:b+1]),
                ("end", ["<s>","<s>"]+toks[b:]+["</s>"])]
        lps = []
        for name, t in segs:
            for i in range(2, len(t)):
                lps.append(self._ngram_logP(self.seg[name], t[i-2], t[i-1], t[i]))
        return float(np.mean(lps)) if lps else -99.0


# ======================================================================
# 구름 기억 생성기 (구름을 '필터'가 아니라 '문맥 기억'으로)
#   RNN의 핵심 아이디어(압축된 문맥 상태를 들고 다님)를 신경망 없이 구름으로.
#   - ctx_vec: 지금까지 문맥의 구름 중심 (기억)
#   - 다음 단어 예측: "비슷한 문맥 다음에 무엇이 왔나"를 코퍼스에서 구름 검색
#   - n-gram(문법) × 구름기억(의미 방향) → 단어 결정
# ======================================================================
class CloudMemoryNgram:
    def __init__(self, sentences, emb, temp_gen=0.6, jm=(0.6, 0.3, 0.1),
                 repeat_penalty=2.0, memory_decay=0.85):
        import math
        from collections import Counter
        from selfloop_engine import tokenize
        self.tok = tokenize; self.emb = emb
        self.temp_gen = temp_gen; self.jm = jm
        self.repeat_penalty = repeat_penalty
        self.memory_decay = memory_decay
        self._math = math
        self.uni = Counter(); self.bi = Counter(); self.tri = Counter()
        self.uni_total = 0; self.vocab = set(); self.word_vec = {}
        # 문맥→다음단어 기억: 각 위치의 (문맥벡터, 다음단어) 쌍
        self.ctx_mem = []      # [(ctx_vec, next_word)]
        import numpy as np
        for s in sentences:
            toks = tokenize(s)
            if len(toks) < 2:
                continue
            for w in toks:
                if w not in self.word_vec:
                    self.word_vec[w] = emb.encode(w)
                self.vocab.add(w)
            t = ["<s>", "<s>"] + toks + ["</s>"]
            for i in range(len(t)):
                self.uni[t[i]] += 1; self.uni_total += 1
                if i >= 1: self.bi[(t[i-1], t[i])] += 1
                if i >= 2: self.tri[(t[i-2], t[i-1], t[i])] += 1
            # 문맥 기억 구축: 누적 문맥벡터 → 다음 단어
            cvec = np.zeros(emb.dim); cn = 0
            for i, w in enumerate(toks):
                wv = self.word_vec[w]
                if cn > 0 and i < len(toks):
                    self.ctx_mem.append((cvec.copy(), w))
                cvec = cvec * self.memory_decay + wv  # 감쇠 누적(최근 강조)
                cn += 1
        self.ctx_mem_vecs = np.array([c for c, _ in self.ctx_mem]) if self.ctx_mem else None
        self.ctx_mem_words = [w for _, w in self.ctx_mem]
        # 정규화 캐시
        if self.ctx_mem_vecs is not None:
            norms = np.linalg.norm(self.ctx_mem_vecs, axis=1, keepdims=True) + 1e-12
            self.ctx_mem_norm = self.ctx_mem_vecs / norms

    def _ngram_logP(self, w1, w2, w3):
        V = max(1, len(self.vocab))
        p_uni = (self.uni.get(w3, 0) + 1) / (self.uni_total + V)
        c_bi = self.uni.get(w2, 0)
        p_bi = (self.bi.get((w2, w3), 0) / c_bi) if c_bi > 0 else 0.0
        c_tri = self.bi.get((w1, w2), 0)
        p_tri = (self.tri.get((w1, w2, w3), 0) / c_tri) if c_tri > 0 else 0.0
        a, b, c = self.jm
        return self._math.log(max(a*p_tri + b*p_bi + c*p_uni, 1e-10))

    def _memory_recall(self, ctx_vec, topk=30):
        """구름 기억 회상: 현재 문맥과 비슷한 과거 문맥들의 '다음 단어' 분포."""
        import numpy as np
        from collections import Counter
        if self.ctx_mem_vecs is None:
            return {}
        cn = ctx_vec / (np.linalg.norm(ctx_vec) + 1e-12)
        sims = self.ctx_mem_norm @ cn
        idx = np.argpartition(-sims, min(topk, len(sims)-1))[:topk]
        # 유사도 가중으로 다음단어 투표
        votes = {}
        for i in idx:
            w = self.ctx_mem_words[i]
            votes[w] = votes.get(w, 0.0) + max(float(sims[i]), 0.0)
        return votes

    def generate(self, max_len=16, seed=0, candidate_pool=120, beta=0.5):
        """
        beta: 구름기억 vs n-gram 비중 (0=n-gram만, 1=기억만)
        매 스텝: n-gram(문법) + 구름기억(의미방향) → 다음 단어.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        cand_words = [w for w, _ in self.uni.most_common(candidate_pool)
                      if w not in ("<s>", "</s>")]
        out = []; w1, w2 = "<s>", "<s>"; recent = {}
        ctx_vec = np.zeros(self.emb.dim); ctx_n = 0
        for _ in range(max_len):
            # 구름 기억 회상
            votes = self._memory_recall(ctx_vec) if ctx_n > 0 else {}
            maxv = max(votes.values()) if votes else 1.0
            scores = []
            for w in cand_words:
                lp = (1 - beta) * self._ngram_logP(w1, w2, w)
                if votes:
                    mem = votes.get(w, 0.0) / maxv  # 0~1 기억 지지
                    lp += beta * np.log(mem + 1e-3)
                if w in recent:
                    lp -= self.repeat_penalty * recent[w]
                scores.append(lp)
            scores = np.array(scores)
            z = scores / max(self.temp_gen, 1e-6); z -= z.max()
            p = np.exp(z); p /= p.sum() + 1e-12
            nxt = cand_words[int(rng.choice(len(cand_words), p=p))]
            if nxt == "</s>":
                break
            out.append(nxt)
            wv = self.word_vec.get(nxt)
            if wv is not None:
                ctx_vec = ctx_vec * self.memory_decay + wv; ctx_n += 1
            recent = {k: v*0.5 for k, v in recent.items()}
            recent[nxt] = recent.get(nxt, 0) + 1
            w1, w2 = w2, nxt
        return " ".join(out)

    def score(self, sentence):
        import numpy as np
        t = ["<s>", "<s>"] + self.tok(sentence) + ["</s>"]
        lps = [self._ngram_logP(t[i-2], t[i-1], t[i]) for i in range(2, len(t))]
        return float(np.mean(lps)) if lps else -99.0


# ======================================================================
# 생성 다양성 측정
#   여러 번 생성한 결과가 서로 얼마나 다른가 (표면 다양성).
# ======================================================================
def diversity_metrics(samples):
    """
    samples: 생성된 문장 리스트
    반환: distinct-1/2(고유 n-gram 비율), self-BLEU 근사(겹침), 어휘 풍부도
    """
    import numpy as np
    from selfloop_engine import tokenize
    toks = [tokenize(s) for s in samples]
    all_uni = [w for t in toks for w in t]
    all_bi = [(t[i], t[i+1]) for t in toks for i in range(len(t)-1)]
    # distinct-n: 고유 n-gram / 전체 n-gram (높을수록 다양)
    d1 = len(set(all_uni)) / max(len(all_uni), 1)
    d2 = len(set(all_bi)) / max(len(all_bi), 1)
    # 문장 간 자카드 겹침 평균 (낮을수록 다양)
    sets = [set(t) for t in toks]
    overlaps = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            u = sets[i] | sets[j]; inter = sets[i] & sets[j]
            overlaps.append(len(inter)/max(len(u), 1))
    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    # 전체 고유 어휘
    vocab_richness = len(set(all_uni))
    return {"distinct1": round(d1, 3), "distinct2": round(d2, 3),
            "avg_pairwise_overlap": round(avg_overlap, 3),
            "unique_vocab": vocab_richness,
            "n_samples": len(samples)}
