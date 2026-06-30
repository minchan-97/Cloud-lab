"""
app_cloud_lab.py — 가스코드 구름 실험실 (Streamlit)

오늘 만든 것들을 한 화면에서 만지며 생각하기 위한 탐구 앱.
  1. 데이터: pkl/텍스트 로드 → 임베딩 + SOM 구름 생성
  2. 구름 관측: 입력이 구름으로 퍼지고 관측 시 붕괴하는가 (temp 조절)
  3. 생성 실험: 구름×n-gram 생성 + 다양성 측정 (alpha 조절)
  4. 호프필드 경계: 정체성 우물 에너지로 영역 안/밖 구분

실행: streamlit run app_cloud_lab.py
"""
import streamlit as st
import numpy as np
import pickle

from selfloop_engine import (EmbeddingProvider, build_tok_emb_from_corpus,
                             GrowingSOM, tokenize, _is_garbage_sentence)
from prob_cloud import (cloud_distribution, cloud_entropy, max_entropy,
                        observe_collapse, participation_ratio, cloud_capacity,
                        GasCloudNgram, diversity_metrics)
from hopfield_boundary import HopfieldBoundary

st.set_page_config(page_title="가스코드 구름 실험실", layout="wide")

# ---------------- 세션 상태 ----------------
for k, v in {"corpus": [], "emb": None, "som_W": None, "gen": None,
             "domain_name": "도메인"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🌫️ 가스코드 구름 실험실")
st.caption("존재하지만 어디 있는지 모른다 · 관측하면 정해진다")

# ===================================================================
# 사이드바: 데이터 로드 & 구름 생성
# ===================================================================
with st.sidebar:
    st.header("1 · 데이터")
    up = st.file_uploader("pkl 또는 txt 업로드", type=["pkl", "txt"])
    dim = st.select_slider("임베딩 차원", [32, 48, 64, 96], value=64)
    epochs = st.slider("임베딩 학습 epoch", 2, 15, 8)
    n_nodes = st.slider("SOM 노드 수", 15, 60, 40)

    if st.button("구름 생성", type="primary", use_container_width=True):
        if up is None:
            st.error("파일을 올려주세요")
        else:
            with st.spinner("임베딩 + SOM 구름 생성 중..."):
                # 코퍼스 추출
                if up.name.endswith(".pkl"):
                    b = pickle.loads(up.read())
                    raw = b.get("corpus", []) if isinstance(b, dict) else []
                else:
                    raw = up.read().decode("utf-8", "ignore").split("\n")
                corpus = [s.strip() for s in set(raw)
                          if s and not _is_garbage_sentence(s) and len(s.strip()) > 10]
                corpus = corpus[:600]
                if len(corpus) < 20:
                    st.error(f"유효 문장이 너무 적습니다 ({len(corpus)})")
                else:
                    emb = EmbeddingProvider(dim=dim)
                    mat, w2i = build_tok_emb_from_corpus(corpus, dim=dim, epochs=epochs)
                    if mat is not None:
                        emb.apply_matrix(mat, w2i)
                    X = emb.encode_many(corpus)
                    n = min(n_nodes, len(X))
                    g = GrowingSOM(dim=dim, init_nodes=n, seed=0)
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(X), n, replace=False)
                    g.W = X[idx].copy() + rng.normal(scale=0.02, size=(n, dim))
                    g.coords = g.coords[:n]; g.err = g.err[:n]
                    for _ in range(30):
                        g.round += 1; g.train_step(X, 0.1, 1.0)
                    st.session_state.corpus = corpus
                    st.session_state.emb = emb
                    st.session_state.som_W = g.W
                    st.session_state.gen = GasCloudNgram(corpus, emb, g.W, temp_gen=0.8)
                    st.success(f"완료: {len(corpus)}문장, {emb.mode}, SOM {n}노드")

    if st.session_state.corpus:
        st.divider()
        st.metric("코퍼스", f"{len(st.session_state.corpus)}문장")
        st.metric("임베딩", st.session_state.emb.mode)
        st.session_state.domain_name = st.text_input("이 영역 이름", st.session_state.domain_name)

# ===================================================================
# 메인: 탭
# ===================================================================
if not st.session_state.corpus:
    st.info("← 사이드바에서 데이터를 올리고 **구름 생성**을 눌러주세요.")
    st.stop()

emb = st.session_state.emb
som_W = st.session_state.som_W
corpus = st.session_state.corpus

tab1, tab2, tab3 = st.tabs(["☁️ 구름 관측", "✨ 생성 + 다양성", "🧲 호프필드 경계"])

# ---------------- 탭1: 구름 관측 ----------------
with tab1:
    st.subheader("입력이 구름으로 퍼지고, 관측하면 붕괴한다")
    q = st.text_input("관측할 문장", corpus[0][:40], key="cloud_q")
    temp = st.slider("관측 온도 (temp) — 낮을수록 입자, 높을수록 구름", 0.05, 5.0, 1.0, 0.05)
    if q.strip():
        x = emb.encode(q)
        p = cloud_distribution(som_W, x, temp=temp)
        ent = cloud_entropy(p)
        pr = participation_ratio(p)
        cidx, conf = observe_collapse(p)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("엔트로피", f"{ent:.2f}", f"최대 {max_entropy(len(p)):.2f}")
        c2.metric("참여 노드", f"{pr:.1f}", f"전체 {len(p)}")
        c3.metric("붕괴 노드", f"#{cidx}")
        c4.metric("붕괴 확신", f"{conf:.2f}")
        state = "입자(확정)" if pr < 2 else ("구름(불확정)" if pr > len(p) * 0.4 else "중간")
        st.info(f"현재 상태: **{state}**")
        st.bar_chart({"확률": p})
        st.caption("temp를 내리면 한 노드로 모이고(관측·붕괴), 올리면 여러 노드로 퍼집니다(중첩·구름).")

# ---------------- 탭2: 생성 + 다양성 ----------------
with tab2:
    st.subheader("구름 × n-gram 생성 — 다양한 재료를 만든다")
    cc1, cc2, cc3 = st.columns(3)
    alpha = cc1.slider("구름 비중 (alpha)", 0.0, 1.0, 0.5, 0.05)
    n_gen = cc2.slider("생성 개수", 5, 30, 15)
    max_len = cc3.slider("문장 길이", 8, 20, 14)
    if st.button("생성 + 다양성 측정", type="primary"):
        gen = st.session_state.gen
        with st.spinner("생성 중..."):
            samples = [gen.generate(max_len=max_len, seed=s * 13 + 1, alpha=alpha)
                       for s in range(n_gen)]
            div = diversity_metrics(samples)
        d1, d2, d3 = st.columns(3)
        d1.metric("distinct-1 (단어 다양성)", div["distinct1"])
        d2.metric("distinct-2 (조합 다양성)", div["distinct2"])
        d3.metric("문장간 겹침", div["avg_pairwise_overlap"], "낮을수록 다양")
        st.caption(f"고유 어휘 {div['unique_vocab']}개 · 샘플 {div['n_samples']}개")
        st.divider()
        st.markdown("**생성된 재료 (거칠어도 됨 — 다양성이 목적, 정제는 LLM 속기사가)**")
        for s in samples:
            st.text(f"· {s}")
        st.info("💡 이 거친 재료들을 마르코프로 선별 → LLM 속기사로 한 번 정제하면 완성. "
                "구름이 다양성을, 마르코프가 품질을, LLM이 매끄러움을 담당.")

# ---------------- 탭3: 호프필드 경계 ----------------
with tab3:
    st.subheader("정체성 우물 — 호프필드 에너지로 영역 안/밖 측정")
    st.caption("현재 코퍼스를 정체성 우물로 박고, 입력이 그 안인지 밖인지 에너지로 판정합니다.")
    beta = st.slider("우물 예민도 (beta)", 1.0, 16.0, 8.0, 0.5)
    n_pat = min(200, len(corpus))
    if "hop" not in st.session_state or st.button("정체성 우물 재구성"):
        P = emb.encode_many(corpus[:n_pat])
        st.session_state.hop = HopfieldBoundary(beta=beta).store(P)
    hop = st.session_state.hop
    hop.beta = beta

    test_q = st.text_area("측정할 문장들 (한 줄에 하나)",
                          "\n".join([corpus[0][:40], corpus[1][:40],
                                     "비트코인 시세가 급등했다", "커피 원두를 볶는 방법"]),
                          height=120)
    if st.button("에너지 측정", type="primary"):
        lines = [l.strip() for l in test_q.split("\n") if l.strip()]
        rows = []
        for s in lines:
            e = hop.energy(emb.encode(s))
            rows.append({"문장": s[:35], "에너지": round(e, 3)})
        rows.sort(key=lambda r: r["에너지"])
        st.markdown("**에너지 낮을수록 정체성 우물 안 (= 이 영역에 속함)**")
        st.table(rows)
        energies = [r["에너지"] for r in rows]
        st.bar_chart({"에너지(낮을수록 영역 안)": energies})
        st.caption("코퍼스 영역의 문장은 낮게, 다른 영역(오염)은 높게 나오면 경계가 작동하는 것. "
                   "단, 영역이 단어를 많이 공유하면(예: 교육 vs 체육) 잘 안 갈립니다.")

st.divider()
st.caption("오늘의 실험 종합 · 구름 표상 / 생성 다양성 / 호프필드 경계 — 만지며 다음을 구상하세요.")
