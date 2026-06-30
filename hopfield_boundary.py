"""
hopfield_boundary.py — 호프필드 에너지로 정체성 경계 측정 (초경량 신경망)

발상 (민찬기): 정체성을 호프필드 우물(attractor)로 박아두고,
들어오는 입력의 에너지로 "정체성에서 얼마나 벗어났나"를 연속 측정.
  - 에너지 낮음 = 정체성 우물 안 (안전)
  - 에너지 높음 = 경계 밖 (오염 위험)
이 에너지를 파인튜닝 안전밸브로 쓴다.

학습(역전파) 없음 — 정체성 패턴을 Hebbian으로 한 번 박고, 에너지만 계산.
연속형 호프필드(실수 벡터) 사용.
"""
from __future__ import annotations
import numpy as np


class HopfieldBoundary:
    def __init__(self, beta=2.0):
        """
        beta: 에너지 날카로움(역온도). 높을수록 우물이 뾰족.
        """
        self.patterns = None      # 정체성 패턴들 (저장된 기억)
        self.beta = beta

    def store(self, patterns):
        """
        정체성 패턴 저장 (Hebbian — 학습 아님, 한 번 박기).
        patterns: (n, dim) 정규화된 정체성 벡터들.
        """
        P = np.asarray(patterns, dtype=np.float64)
        norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        self.patterns = P / norms
        return self

    def energy(self, x):
        """
        입력 x의 호프필드 에너지 (modern Hopfield, log-sum-exp 형태).
        정체성 패턴과 가까울수록(우물 안) 에너지 낮음.
        E(x) = -1/beta * logsumexp(beta * <patterns, x>)
        """
        xv = np.asarray(x, dtype=np.float64)
        xv = xv / (np.linalg.norm(xv) + 1e-12)
        sims = self.patterns @ xv              # 각 정체성 패턴과의 정합
        m = np.max(self.beta * sims)
        lse = m + np.log(np.sum(np.exp(self.beta * sims - m)) + 1e-12)
        return float(-lse / self.beta)

    def retrieve(self, x, steps=1):
        """
        복원: 입력을 정체성 우물로 한 스텝 끌어당김 (modern Hopfield update).
        오염된 입력을 정체성 방향으로 교정.
        """
        xv = np.asarray(x, dtype=np.float64)
        xv = xv / (np.linalg.norm(xv) + 1e-12)
        for _ in range(steps):
            sims = self.patterns @ xv
            w = np.exp(self.beta * sims - np.max(self.beta * sims))
            w /= w.sum() + 1e-12
            xv = w @ self.patterns               # 패턴들의 가중합으로 이동
            xv = xv / (np.linalg.norm(xv) + 1e-12)
        return xv
