# =====================================================================
mdfig(r"""
---
# 6. Filtro de Kalman estendido iterado (IEKF)

![Figura 07 do original: o ciclo do IEKF (veja a nota do tradutor sobre a caixa "correction2")](attachment:07.png)

O IEKF é o EKF com a etapa de correção **repetida**. Como o EKF lineariza funções não lineares, a linearização sempre introduz erro. O IEKF reduz esse erro **linearizando de novo** em torno da estimativa corrigida, enquanto a mudança $\delta\hat{\mathbf x}_{t|t,j}$ produzida pela correção for grande.

Aqui, $\delta\hat{\mathbf x}$ é interpretado como **incremento de atualização**, e não como estado de erro: serve apenas para passar da $j$-ésima estimativa *a posteriori* para a $(j+1)$-ésima, e não é ele mesmo objeto de estimação.

## 6.1 Comparação com o EKF

**Semelhança 1.** Os modelos de movimento e de observação são exatamente os do EKF: $\mathbf x_t=\mathbf f(\mathbf x_{t-1},\mathbf u_t,\mathbf w_t)$ e $\mathbf z_t=\mathbf h(\mathbf x_t,\mathbf v_t)$.

**Semelhança 2.** A linearização também é a do EKF: $\mathbf x_t=\mathbf F_t\mathbf x_{t-1}+\tilde{\mathbf u}_t+\tilde{\mathbf w}_t$ e $\mathbf z_t=\mathbf H_t\mathbf x_t+\tilde{\mathbf z}_t+\tilde{\mathbf v}_t$, com $\tilde{\mathbf z}_t=\mathbf h(\hat{\mathbf x}_{t|t-1},0)-\mathbf H_t\hat{\mathbf x}_{t|t-1}$.

**Semelhança 3.** Como no EKF, as matrizes $\mathbf F,\mathbf H,\mathbf K$ se referem ao estado verdadeiro (e não a um estado de erro).

**Diferença 1.** No EKF, a correção vai da predição ao valor corrigido em **um** passo, $\hat{\mathbf x}_{t|t-1}\to\hat{\mathbf x}_{t|t}$. No IEKF, o valor corrigido volta a ser usado como ponto de linearização, e a correção se repete, $\hat{\mathbf x}_{t|t,j}\rightleftarrows\hat{\mathbf x}_{t|t,j+1}$ ($j$ é o número da iteração).

**Diferença 2.** Escrevendo a inovação **linearizada em torno de um ponto qualquer** $\hat{\mathbf x}_{t|t,j}$,

$$\mathbf r_{t,j}=\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0)-\mathbf H_{t,j}(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j}).$$

- **EKF:** o ponto de linearização é a própria predição, $\hat{\mathbf x}_{t|t,0}=\hat{\mathbf x}_{t|t-1}$; o último termo some, e sobra a inovação usual $\mathbf r_t=\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t-1},0)$.
- **IEKF:** a cada iteração lineariza-se num ponto novo, e o termo **não** some a partir de $j=1$.

O último termo "traduz" a medida prevista pela linearização em $\hat{\mathbf x}_{t|t,j}$ de volta para a predição $\hat{\mathbf x}_{t|t-1}$: **é a predição que está sendo corrigida**, e não o iterado anterior.

## 6.2 Predição (igual à do EKF)

$$\boxed{\hat{\mathbf x}_{t|t-1}=\mathbf f(\hat{\mathbf x}_{t-1|t-1},\mathbf u_t,0),\qquad \mathbf P_{t|t-1}=\mathbf F_t\mathbf P_{t-1|t-1}\mathbf F_t^\top+\mathbf F_{\mathbf w}\mathbf Q_t\mathbf F_{\mathbf w}^\top}$$

## 6.3 Correção (forma corrigida)

$$\boxed{\begin{aligned}
&\hat{\mathbf x}_{t|t,0}=\hat{\mathbf x}_{t|t-1};\ \text{escolha } \epsilon\\
&\text{para } j=0,1,2,\dots\\
&\qquad \mathbf K_{t,j}=\mathbf P_{t|t-1}\mathbf H_{t,j}^\top(\mathbf H_{t,j}\mathbf P_{t|t-1}\mathbf H_{t,j}^\top+\mathbf H_{\mathbf v,j}\mathbf R_t\mathbf H_{\mathbf v,j}^\top)^{-1}\\
&\qquad \hat{\mathbf x}_{t|t,j+1}={\color{red}\hat{\mathbf x}_{t|t-1}}+\mathbf K_{t,j}\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0)-\mathbf H_{t,j}(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j})\big)\\
&\qquad \text{pare quando } \|\hat{\mathbf x}_{t|t,j+1}-\hat{\mathbf x}_{t|t,j}\|<\epsilon\\
&\mathbf P_{t|t}=(\mathbf I-\mathbf K_{t,n}\mathbf H_{t,n})\mathbf P_{t|t-1}\quad(\text{com o último } j=n)
\end{aligned}}$$

> **Nota do tradutor (a mais importante deste notebook).** No original, a correção do IEKF é escrita como
> $$\delta\hat{\mathbf x}_{t|t,j}=\mathbf K_{t,j}\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0)-\mathbf H_t(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j})\big),\qquad \hat{\mathbf x}_{t|t,j+1}={\color{red}\hat{\mathbf x}_{t|t,j}}+\delta\hat{\mathbf x}_{t|t,j},$$
> isto é, o incremento é somado ao **iterado anterior**, e não à **predição**. A mesma forma aparece no quadro-resumo e nas figuras 07 e 12. **Assim escrito, o algoritmo não converge**: o termo $-\mathbf H(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j})$ já "traz a correção de volta" para a predição, e somá-lo ao iterado conta a correção duas vezes, três vezes, e assim por diante. No caso **linear**, $\mathbf r_{t,j}=\mathbf z_t-\mathbf H\hat{\mathbf x}_{t|t-1}$ não depende de $j$, e a forma impressa anda a mesma distância a cada iteração, para sempre.
>
> A forma correta é a de Bell e Cathey (1993), em que a soma é feita à predição $\hat{\mathbf x}_{t|t-1}$. É também a forma que **o próprio autor deduz** na seção 9 (última linha da dedução por Gauss–Newton) e a que se obtém da fórmula do IESKF da seção 7 com $\mathbf J=\mathbf I$. Também está correta, nos dois sentidos, a escrita "incremental": $\hat{\mathbf x}_{t|t,j+1}=\hat{\mathbf x}_{t|t,j}+\big[\mathbf K_{t,j}\mathbf r_{t,j}-(\hat{\mathbf x}_{t|t,j}-\hat{\mathbf x}_{t|t-1})\big]$: o termo entre colchetes é o incremento verdadeiro, e é exatamente o que falta no original. (Na equação da seção 9 o original ainda usa $\mathbf H_t$ fixo, sem o índice $j$; ele deve ser recalculado a cada iteração, como acima.)

A célula abaixo implementa as duas formas e as aplica primeiro a um caso **linear** e depois a um caso não linear, uma medida de distância e direção muito precisa feita de perto, com uma predição muito incerta. Nesse cenário, o EKF erra feio.
""", "07.png")

code(r"""
def iekf_correcao(x_pred, P, z, h, Hfun, R, forma="correta", n_iter=30, eps=1e-10):
    # IEKF. forma="correta": soma à predição (Bell–Cathey); "impressa": soma ao iterado anterior.
    xj = x_pred.copy(); iterados = [xj.copy()]
    for j in range(n_iter):
        H = Hfun(xj)
        K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
        r = z - h(xj) - H @ (x_pred - xj)
        x_novo = (x_pred if forma == "correta" else xj) + K @ r
        iterados.append(x_novo.copy())
        parou = np.linalg.norm(x_novo - xj) < eps
        xj = x_novo
        if parou:
            break
    H = Hfun(xj); K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
    return xj, (np.eye(len(xj)) - K @ H) @ P, np.array(iterados)

# --- caso linear: o IEKF correto pára na 1ª iteração; o impresso anda sem parar
H_lin = np.array([[1.0, 0.0]]); P_lin = np.eye(2); R_lin = np.array([[0.1]])
xp_lin, z_lin = np.zeros(2), np.array([1.0])
xc, _, it_c = iekf_correcao(xp_lin, P_lin, z_lin, lambda x: H_lin @ x, lambda x: H_lin, R_lin, "correta", n_iter=6)
xi, _, it_i = iekf_correcao(xp_lin, P_lin, z_lin, lambda x: H_lin @ x, lambda x: H_lin, R_lin, "impressa", n_iter=6, eps=0)
x_kf_lin, _, _ = kf_correcao(xp_lin, P_lin, z_lin, H_lin, R_lin)
print("caso linear, primeira componente dos iterados")
print("  forma correta :", np.round(it_c[:, 0], 4))
print("  forma impressa:", np.round(it_i[:, 0], 4))
confere("caso linear: IEKF correto = KF (uma correção basta)", xc, x_kf_lin, tol=1e-12)
""")

code(r"""
def h_rb(p):                     # distância e direção, sensor na origem
    return np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])])

def H_rb(p):
    r2 = p[0]**2 + p[1]**2; r = np.sqrt(r2)
    return np.array([[p[0] / r, p[1] / r], [-p[1] / r2, p[0] / r2]])

xp_rb = np.array([3.0, 0.0]); P_rb = np.diag([1.0, 4.0])          # predição incerta, sobretudo em y
R_rb = np.diag([0.05**2, 0.02**2])                                 # sensor muito bom
x_verd_rb = np.array([2.2, 1.6]); z_rb = h_rb(x_verd_rb)           # medida sem ruído, para isolar a linearização

x_c, P_c, it_c = iekf_correcao(xp_rb, P_rb, z_rb, h_rb, H_rb, R_rb, "correta")
x_i, _, it_i = iekf_correcao(xp_rb, P_rb, z_rb, h_rb, H_rb, R_rb, "impressa", n_iter=12, eps=0)

# MAP por otimização direta (a resposta que o IEKF deveria encontrar)
Lr, Lp = np.linalg.cholesky(np.linalg.inv(R_rb)).T, np.linalg.cholesky(np.linalg.inv(P_rb)).T
x_map = least_squares(lambda p: np.concatenate([Lr @ (z_rb - h_rb(p)), Lp @ (p - xp_rb)]), xp_rb,
                      xtol=1e-15, ftol=1e-15, gtol=1e-15).x
confere("IEKF (forma correta) converge para o MAP", x_c, x_map, tol=1e-9)

tab = pd.DataFrame({"forma correta: |x_j − MAP|": np.linalg.norm(it_c[:13] - x_map, axis=1) if len(it_c) >= 13 else
                    np.r_[np.linalg.norm(it_c - x_map, axis=1), [np.nan] * (13 - len(it_c))],
                    "forma impressa: |x_j − MAP|": np.linalg.norm(it_i[:13] - x_map, axis=1)})
tab.index.name = "iteração j"

th = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(figsize=(7, 6))
for nome, Pm, xm, cor in [("predição", P_rb, xp_rb, "gray")]:
    L = np.linalg.cholesky(Pm); ax.plot(*(xm[:, None] + 2 * L @ np.vstack([np.cos(th), np.sin(th)])), color=cor, ls="--", label="predição (2σ)")
ax.plot(*np.vstack([z_rb[0] * np.cos(th), z_rb[0] * np.sin(th)]), color="tab:blue", lw=.6, alpha=.5, label="lugar da medida de distância")
ax.plot(it_c[:, 0], it_c[:, 1], "o-", color="tab:green", label="IEKF, forma correta")
ax.plot(it_i[:, 0], it_i[:, 1], "s-", color="tab:red", ms=4, label="IEKF, forma impressa no original")
ax.plot(*x_map, "k*", ms=16, label="MAP (otimização direta)")
ax.annotate("j = 1 (EKF)", it_c[1], xytext=(10, -15), textcoords="offset points", fontsize=8)
ax.set_xlim(-1, 11); ax.set_ylim(-4, 11); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("Iterações do IEKF: a forma correta chega ao MAP; a impressa vai embora")
plt.tight_layout(); plt.show()
tab.round(4)
""")

md(r"""
**Leitura.**
- A **primeira** iteração das duas formas é a mesma, e é o **EKF**: um ponto bem longe do MAP, porque a linearização em $\hat{\mathbf x}_{t|t-1}=(3,0)$ é ruim para uma predição tão incerta.
- A forma correta chega ao MAP em poucas iterações, com a convergência rápida típica de Gauss–Newton.
- A forma impressa se afasta a cada iteração. No caso linear, ela anda a mesma distância, $\mathbf K(\mathbf z-\mathbf H\hat{\mathbf x}_{t|t-1})$, em toda iteração.

Esta é a mesma "caça ao erro" da seção C5b de `mq_hmm_kalman_duas_trilhas`, agora feita no texto original completo, com as figuras.

## 6.4 Resumo (forma corrigida)

$$
\boxed{\begin{aligned}
&\text{FiltroDeKalmanEstendidoIterado}(\hat{\mathbf x}_{t-1|t-1},\mathbf P_{t-1|t-1},\mathbf u_t,\mathbf z_t)\ \{\\
&\qquad\text{(predição)}\quad \text{igual à do EKF}\\
&\qquad\text{(correção)}\quad \hat{\mathbf x}_{t|t,0}=\hat{\mathbf x}_{t|t-1};\ \text{repita: } \mathbf K_{t,j}=\mathbf P_{t|t-1}\mathbf H_{t,j}^\top(\mathbf H_{t,j}\mathbf P_{t|t-1}\mathbf H_{t,j}^\top+\mathbf H_{\mathbf v,j}\mathbf R_t\mathbf H_{\mathbf v,j}^\top)^{-1},\\
&\qquad\qquad\qquad\qquad \hat{\mathbf x}_{t|t,j+1}=\hat{\mathbf x}_{t|t-1}+\mathbf K_{t,j}\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0)-\mathbf H_{t,j}(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j})\big)\ \text{ até convergir}\\
&\qquad\qquad\qquad\qquad \mathbf P_{t|t}=(\mathbf I-\mathbf K_{t,n}\mathbf H_{t,n})\mathbf P_{t|t-1}\\
&\qquad\text{retorna } \hat{\mathbf x}_{t|t},\ \mathbf P_{t|t}\ \}
\end{aligned}}
$$
""")

# =====================================================================
mdfig(r"""
---
# 7. Filtro de Kalman de estado de erro iterado (IESKF)

![Figura 08 do original: o ciclo do IESKF](attachment:08.png)

Enquanto o IEKF estima iterativamente o estado verdadeiro na correção, o **IESKF** estima iterativamente o **estado de erro** $\delta\mathbf x_t$. Vale a mesma relação $\mathbf x_{\text{verd},t}=\mathbf x_t\boxplus\delta\mathbf x_t$. Na atualização iterativa, pode-se entender $\mathbf x_{\text{verd},t}$ como o estado **depois** da atualização, o nominal $\mathbf x_t$ como o estado **antes**, e o erro $\delta\mathbf x_t$ como o **tamanho** da atualização.

## 7.1 Comparação com o ESKF

**Semelhança 1.** Os modelos são os do ESKF.

**Semelhança 2.** Os jacobianos $\mathbf F_t$ e $\mathbf H_t$ são em relação ao erro $\delta\mathbf x_t$, como no ESKF, e $\mathbf H_t=\frac{\partial\mathbf h}{\partial\mathbf x_{\text{verd},t}}\frac{\partial\mathbf x_{\text{verd},t}}{\partial\delta\mathbf x_t}$.

**Semelhança 3.** O modelo linearizado do erro é o mesmo do ESKF: $\delta\mathbf x_t=\mathbf F_t\delta\mathbf x_{t-1}+\tilde{\mathbf w}_t$.

**Diferença: a correção é deduzida como um problema MAP.** A correção do ESKF desenvolve $\mathrm{bel}(\delta\mathbf x_t)=\eta\,p(\mathbf z_t\mid\delta\mathbf x_t)\,\overline{\mathrm{bel}}(\delta\mathbf x_t)$ e chega a $\delta\hat{\mathbf x}_{t|t}=\mathbf K_t(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t-1},0,0))$ e $\mathbf P_{t|t}=(\mathbf I-\mathbf K_t\mathbf H_t)\mathbf P_{t|t-1}$. O IESKF, em vez disso, usa a **estimação de máximo *a posteriori*** (MAP), resolvida por **Gauss–Newton** (seção 9). A dedução segue as referências [6]–[11] do original, das quais [6] (He, Xu e Zhang, *Kalman filters on differentiable manifolds*) é a mais detalhada.

Se o EKF é deduzido pela via MAP, o problema a otimizar é

$$\arg\min_{\mathbf x_t}\ \big\|\mathbf z_t-\mathbf h(\mathbf x_{\text{verd},t},0)\big\|^2_{(\mathbf H_{\mathbf v}\mathbf R\mathbf H_{\mathbf v}^\top)^{-1}}+\big\|\mathbf x_{\text{verd},t}\boxminus\hat{\mathbf x}_{t|t-1}\big\|^2_{\mathbf P_{t|t-1}^{-1}},$$

com $\|\mathbf a\|^2_{\mathbf W}=\mathbf a^\top\mathbf W\mathbf a$.

> **Nota do tradutor.** O original escreve o peso do primeiro termo como $\mathbf H_{\mathbf v}\mathbf R^{-1}\mathbf H_{\mathbf v}^\top$ (aqui e nas seções 7 e 10). O peso correto é o **inverso da covariância** do ruído linearizado, $(\mathbf H_{\mathbf v}\mathbf R\mathbf H_{\mathbf v}^\top)^{-1}$; as duas expressões só coincidem quando $\mathbf H_{\mathbf v}$ é ortogonal (por exemplo, $\mathbf H_{\mathbf v}=\mathbf I$).

> **Dica (do original).** O erro **de predição** é $\delta\hat{\mathbf x}_{t|t-1}=\mathbf x_{\text{verd},t}\boxminus\hat{\mathbf x}_{t|t-1}\sim\mathcal N(0,\mathbf P_{t|t-1})$. O erro **corrigido** é $\delta\hat{\mathbf x}_{t|t}=\mathbf x_{\text{verd},t}\boxminus\hat{\mathbf x}_{t|t}$, ou seja, $\mathbf x_{\text{verd},t}=\hat{\mathbf x}_{t|t}\boxplus\delta\hat{\mathbf x}_{t|t}$.

O primeiro termo é linearizado como no ESKF, **mas agora em torno do iterado atual** $\hat{\mathbf x}_{t|t,j}$:

$$\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j}\boxplus\delta\mathbf x,0)\approx\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0)-\mathbf H_{t,j}\,\delta\mathbf x .$$

No ESKF, o erro estimado é sempre zero e esse termo desaparecia; no IESKF ele não é zero e fica. O segundo termo é desenvolvido usando a dica:

$$\delta\hat{\mathbf x}_{t|t-1}=(\hat{\mathbf x}_{t|t,j}\boxplus\delta\mathbf x)\boxminus\hat{\mathbf x}_{t|t-1}\approx(\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1})+\mathbf J_{t,j}\,\delta\mathbf x,\qquad \mathbf J_{t,j}=\frac{\partial}{\partial\delta\mathbf x}\Big((\hat{\mathbf x}_{t|t,j}\boxplus\delta\mathbf x)\boxminus\hat{\mathbf x}_{t|t-1}\Big)\Big|_{\delta\mathbf x=0}.$$

Daí, como $\delta\hat{\mathbf x}_{t|t-1}\sim\mathcal N(0,\mathbf P_{t|t-1})$, o erro corrigido tem distribuição *a priori*

$$\delta\mathbf x\sim\mathcal N\big(-\mathbf J_{t,j}^{-1}(\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1}),\ \mathbf J_{t,j}^{-1}\mathbf P_{t|t-1}\mathbf J_{t,j}^{-\top}\big),$$

e o problema MAP da $j$-ésima iteração fica

$$\boxed{\arg\min_{\delta\mathbf x}\ \big\|\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0)-\mathbf H_{t,j}\delta\mathbf x\big\|^2_{(\mathbf H_{\mathbf v}\mathbf R\mathbf H_{\mathbf v}^\top)^{-1}}+\big\|(\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1})+\mathbf J_{t,j}\delta\mathbf x\big\|^2_{\mathbf P_{t|t-1}^{-1}}}$$

> **Nota do tradutor.** No original, o primeiro termo desse problema aparece com $\mathbf h(\hat{\mathbf x}_{t|t-1},0,0)$; mas a fórmula final (e a implementação correta) usa $\mathbf h(\hat{\mathbf x}_{t|t,j},0,0)$, isto é, relineariza no iterado. Também a "soma" $\hat{\mathbf x}_{t|t,j}-\hat{\mathbf x}_{t|t-1}$ deve ser lida como $\boxminus$ quando há rotações.

## 7.2 Predição (igual à do ESKF)

$$\boxed{\delta\hat{\mathbf x}_{t|t-1}=0,\qquad \hat{\mathbf x}_{t|t-1}=\mathbf f(\hat{\mathbf x}_{t-1|t-1},0,\mathbf u_t,0),\qquad \mathbf P_{t|t-1}=\mathbf F_t\mathbf P_{t-1|t-1}\mathbf F_t^\top+\mathbf F_{\mathbf w}\mathbf Q_t\mathbf F_{\mathbf w}^\top}$$

## 7.3 Correção

Derivando o problema acima e igualando a zero (seção 10), com $\mathbf c_j=\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1}$:

$$\boxed{\begin{aligned}
&\hat{\mathbf x}_{t|t,0}=\hat{\mathbf x}_{t|t-1};\ \text{para } j=0,1,\dots\\
&\qquad \mathbf S_{t,j}=\mathbf H_{t,j}\mathbf J_{t,j}^{-1}\mathbf P_{t|t-1}\mathbf J_{t,j}^{-\top}\mathbf H_{t,j}^\top+\mathbf H_{\mathbf v,j}\mathbf R_t\mathbf H_{\mathbf v,j}^\top\\
&\qquad \mathbf K_{t,j}=\mathbf J_{t,j}^{-1}\mathbf P_{t|t-1}\mathbf J_{t,j}^{-\top}\mathbf H_{t,j}^\top\mathbf S_{t,j}^{-1}\\
&\qquad \delta\hat{\mathbf x}_{t|t,j}=\mathbf K_{t,j}\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0,0)+\mathbf H_{t,j}\mathbf J_{t,j}^{-1}\mathbf c_j\big)-\mathbf J_{t,j}^{-1}\mathbf c_j\\
&\qquad \hat{\mathbf x}_{t|t,j+1}=\hat{\mathbf x}_{t|t,j}\boxplus\delta\hat{\mathbf x}_{t|t,j}\qquad\text{até } \|\delta\hat{\mathbf x}_{t|t,j}\|<\epsilon\\
&\mathbf P_{t|t}=(\mathbf I-\mathbf K_{t,n}\mathbf H_{t,n})\mathbf J_{t,n}^{-1}\mathbf P_{t|t-1}\mathbf J_{t,n}^{-\top}
\end{aligned}}$$

**Repare no último termo, $-\mathbf J^{-1}\mathbf c_j$.** É justamente o termo que falta na correção do IEKF impressa no original (seção 6). Com estados vetoriais, $\boxplus=+$, $\boxminus=-$ e $\mathbf J=\mathbf I$, a fórmula do IESKF fica

$$\hat{\mathbf x}_{t|t,j+1}=\hat{\mathbf x}_{t|t,j}+\mathbf K(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j})+\mathbf H\mathbf c_j)-\mathbf c_j=\hat{\mathbf x}_{t|t-1}+\mathbf K\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j})-\mathbf H(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j})\big),$$

que é o IEKF **correto**. Ou seja: o mesmo texto traz, na seção 7, a fórmula que corrige a seção 6.
""", "08.png")

code(r"""
def ieskf_correcao(x_pred, P, z, h, Hfun, R, boxplus, boxminus, Jfun, n_iter=50, eps=1e-12):
    # IESKF genérico: o estado pode viver numa variedade (⊞, ⊟); J é o jacobiano de (x_j ⊞ δ) ⊟ x_pred
    xj = x_pred.copy() if hasattr(x_pred, "copy") else x_pred
    hist = [xj]
    for j in range(n_iter):
        c = boxminus(xj, x_pred)
        Ji = np.linalg.inv(Jfun(c))
        H = Hfun(xj)
        Pb = Ji @ P @ Ji.T
        S = H @ Pb @ H.T + R
        K = Pb @ H.T @ np.linalg.inv(S)
        dx = K @ (z - h(xj) + H @ Ji @ c) - Ji @ c
        xj = boxplus(xj, dx); hist.append(xj)
        if np.linalg.norm(dx) < eps:
            break
    c = boxminus(xj, x_pred); Ji = np.linalg.inv(Jfun(c)); H = Hfun(xj)
    Pb = Ji @ P @ Ji.T; K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
    return xj, (np.eye(P.shape[0]) - K @ H) @ Pb, hist

# espaço vetorial: ⊞ = +, ⊟ = −, J = I  ⇒  deve reproduzir o IEKF correto, iterado a iterado
x_ie, P_ie, hist_ie = ieskf_correcao(xp_rb, P_rb, z_rb, h_rb, H_rb, R_rb,
                                     lambda x, d: x + d, lambda a, b: a - b, lambda c: np.eye(2))
n = min(len(hist_ie), len(it_c))
confere("IESKF com ⊞ = + e J = I reproduz os iterados do IEKF correto", np.array(hist_ie[:n]), it_c[:n], tol=1e-9)
confere("e a mesma covariância final", P_ie, P_c, tol=1e-12)
""")

md(r"""
### Acréscimo: o IESKF na atitude, onde $\mathbf J\neq\mathbf I$

Voltamos à atitude da seção 5.4, agora com **uma única** correção difícil: a predição está errada por cerca de 40° e é muito incerta ($\mathbf P=(35°)^2\mathbf I$), e o par acelerômetro–magnetômetro é preciso. Para rotações, com $\mathbf R\boxminus\mathbf R'=\mathrm{Log}(\mathbf R'^\top\mathbf R)$,

$$\mathbf J_{t,j}=\frac{\partial}{\partial\delta\boldsymbol\theta}\mathrm{Log}\big(\hat{\mathbf R}_{t|t-1}^\top\hat{\mathbf R}_{t|t,j}\mathrm{Exp}(\delta\boldsymbol\theta)\big)\Big|_0=\mathbf J_r^{-1}(\mathbf c_j),\qquad \mathbf c_j=\hat{\mathbf R}_{t|t,j}\boxminus\hat{\mathbf R}_{t|t-1}.$$

Comparamos três respostas:
1. o **ESKF**, que faz uma só correção (é a iteração $j=0$ do IESKF);
2. o **IESKF**, iterado até convergir;
3. o **MAP exato na variedade**, obtido por otimização direta em $\delta\boldsymbol\theta$ com $\mathbf R=\hat{\mathbf R}_{t|t-1}\mathrm{Exp}(\delta\boldsymbol\theta)$.
""")

code(r"""
J_teste = Jr_inv(np.array([0.3, -0.5, 0.2]))
R_j, R_p = Exp(np.array([0.3, -0.5, 0.2])), np.eye(3)
confere("J = J_r⁻¹(c) (numérico)", J_teste,
        jacobiano_numerico(lambda d: boxminus(boxplus(R_j, d), R_p), np.zeros(3)), tol=1e-7)

R_verd1 = Exp(np.array([0.4, 1.1, -0.7]))
R_pred1 = R_verd1 @ Exp(np.radians([30.0, -20.0, 20.0]))     # predição errada por ~40°
P_pred1 = np.radians(35.0)**2 * np.eye(3)
R_med1 = np.diag([0.05**2] * 3 + [0.005**2] * 3)
z1 = h_att(R_verd1) + rng.multivariate_normal(np.zeros(6), R_med1)

R_ie, P_ie1, hist_att = ieskf_correcao(R_pred1, P_pred1, z1, h_att, H_att, R_med1, boxplus, boxminus, Jr_inv)

# MAP exato na variedade
Lr1, Lp1 = np.linalg.cholesky(np.linalg.inv(R_med1)).T, np.linalg.cholesky(np.linalg.inv(P_pred1)).T
sol = least_squares(lambda d: np.concatenate([Lr1 @ (z1 - h_att(boxplus(R_pred1, d))), Lp1 @ d]),
                    np.zeros(3), xtol=1e-15, ftol=1e-15, gtol=1e-15)
R_map1 = boxplus(R_pred1, sol.x)
confere("IESKF na atitude converge para o MAP na variedade", boxminus(R_ie, R_map1), np.zeros(3), tol=1e-7)

graus = lambda R: np.degrees(np.linalg.norm(boxminus(R, R_verd1)))
tab = pd.DataFrame({"erro angular até a verdade (°)": [graus(R_pred1), graus(hist_att[1]), graus(R_ie), graus(R_map1)],
                    "distância até o MAP (°)": [np.degrees(np.linalg.norm(boxminus(R, R_map1))) for R in [R_pred1, hist_att[1], R_ie, R_map1]]},
                   index=["predição", "ESKF (uma correção, j = 1)", f"IESKF ({len(hist_att) - 1} iterações)", "MAP na variedade"])
print("distância ao MAP ao longo das iterações (graus):",
      np.round([np.degrees(np.linalg.norm(boxminus(R, R_map1))) for R in hist_att[:8]], 5))
tab.round(4)
""")

md(r"""
**Leitura.**
- Uma correção só (ESKF) melhora muito a predição, mas deixa um erro grande de linearização. O IESKF elimina esse erro e chega ao MAP exato **na variedade**.
- A convergência é rápida: a distância ao MAP cai de cerca de 41° para 6°, 0,25°, 0,002°, ou seja, o número de casas decimais corretas praticamente dobra a cada iteração, como em Gauss–Newton com resíduo pequeno. Na prática, 3 a 5 iterações bastam.
- É exatamente esta correção iterada na variedade que faz funcionar odometrias como a FAST-LIO (referência [8] do original), com o estado de um robô inteiro (posição, velocidade, rotação e vieses dos sensores) no lugar da nossa rotação.

## 7.4 Resumo

$$
\boxed{\begin{aligned}
&\text{FiltroDeKalmanDeEstadoDeErroIterado}(\hat{\mathbf x}_{t-1|t-1},\mathbf P_{t-1|t-1},\mathbf u_t,\mathbf z_t)\ \{\\
&\qquad\text{(predição)}\quad \text{igual à do ESKF}\\
&\qquad\text{(correção)}\quad \text{repita: } \mathbf S_{t,j},\ \mathbf K_{t,j},\ \delta\hat{\mathbf x}_{t|t,j}\ \text{como na seção 7.3},\ \ \hat{\mathbf x}_{t|t,j+1}=\hat{\mathbf x}_{t|t,j}\boxplus\delta\hat{\mathbf x}_{t|t,j}\ \text{ até convergir}\\
&\qquad\qquad\qquad\qquad \mathbf P_{t|t}=(\mathbf I-\mathbf K_{t,n}\mathbf H_{t,n})\mathbf J_{t,n}^{-1}\mathbf P_{t|t-1}\mathbf J_{t,n}^{-\top}\\
&\qquad\text{retorna } \hat{\mathbf x}_{t|t},\ \mathbf P_{t|t}\ \}
\end{aligned}}
$$

> **Nota do tradutor.** Como no ESKF, o quadro-resumo do original escreve a predição do erro como $\mathbf F_t\hat{\mathbf x}_{t-1|t-1}$ em vez de $\mathbf F_t\,\delta\hat{\mathbf x}_{t-1|t-1}$.
""")

# =====================================================================
md(r"""
---
# 8. Dedução do filtro de Kalman

Esta seção deduz as equações de predição e de atualização do filtro de Kalman, seguindo a seção 3.2.4, "Mathematical Derivation of the KF", de *Probabilistic Robotics* (Thrun, Burgard e Fox, referência [13] do original). O KF supõe que $\mathrm{bel}(\mathbf x_t)$ e $\overline{\mathrm{bel}}(\mathbf x_t)$ são gaussianas, e calcula a média e a covariância de cada uma a partir de

$$\overline{\mathrm{bel}}(\mathbf x_t)=\int{\color{red}p(\mathbf x_t\mid\mathbf x_{t-1},\mathbf u_t)}\,\mathrm{bel}(\mathbf x_{t-1})\,d\mathbf x_{t-1},\qquad \mathrm{bel}(\mathbf x_t)=\eta\,{\color{teal}p(\mathbf z_t\mid\mathbf x_t)}\,\overline{\mathrm{bel}}(\mathbf x_t),$$

com ${\color{red}p(\mathbf x_t\mid\mathbf x_{t-1},\mathbf u_t)}=\mathcal N(\mathbf F_t\mathbf x_{t-1}+\mathbf B_t\mathbf u_t,\mathbf Q_t)$ e ${\color{teal}p(\mathbf z_t\mid\mathbf x_t)}=\mathcal N(\mathbf H_t\mathbf x_t,\mathbf R_t)$.

> **Nota do tradutor: troca de nomes.** Em toda esta seção (e nas figuras 02, 04 e 25), o original usa a convenção de *Probabilistic Robotics*: **$\mathbf R_t$ para o ruído de movimento e $\mathbf Q_t$ para o ruído de medida**, o contrário do resto do texto. A tradução abaixo usa a convenção do resto do notebook ($\mathbf Q$ movimento, $\mathbf R$ medida). Quem ler o original lado a lado deve trocar as letras.

## 8.1 Dedução da predição

Queremos a média e a covariância de $\overline{\mathrm{bel}}(\mathbf x_t)$. Escrevendo as gaussianas,

$$\overline{\mathrm{bel}}(\mathbf x_t)=\eta\int\exp(-\mathbf L_t)\,d\mathbf x_{t-1},\qquad \boxed{\mathbf L_t=\tfrac12(\mathbf x_t-\mathbf F_t\mathbf x_{t-1}-\mathbf B_t\mathbf u_t)^\top\mathbf Q_t^{-1}(\mathbf x_t-\mathbf F_t\mathbf x_{t-1}-\mathbf B_t\mathbf u_t)+\tfrac12(\mathbf x_{t-1}-\hat{\mathbf x}_{t-1})^\top\mathbf P_{t-1}^{-1}(\mathbf x_{t-1}-\hat{\mathbf x}_{t-1})}$$

$\mathbf L_t$ é quadrática em $\mathbf x_t$ e $\mathbf x_{t-1}$. Para tirar a integral, separamos $\mathbf L_t=\mathbf L_t(\mathbf x_{t-1},\mathbf x_t)+\mathbf L_t(\mathbf x_t)$: a primeira parte contém todos os termos com $\mathbf x_{t-1}$; a segunda, só termos em $\mathbf x_t$, e por isso sai da integral:

$$\overline{\mathrm{bel}}(\mathbf x_t)=\eta\,\exp(-\mathbf L_t(\mathbf x_t))\int\exp(-\mathbf L_t(\mathbf x_{t-1},\mathbf x_t))\,d\mathbf x_{t-1}.$$

Para construir $\mathbf L_t(\mathbf x_{t-1},\mathbf x_t)$, derivamos $\mathbf L_t$ em relação a $\mathbf x_{t-1}$:

$$\frac{\partial\mathbf L_t}{\partial\mathbf x_{t-1}}=-\mathbf F_t^\top\mathbf Q_t^{-1}(\mathbf x_t-\mathbf F_t\mathbf x_{t-1}-\mathbf B_t\mathbf u_t)+\mathbf P_{t-1}^{-1}(\mathbf x_{t-1}-\hat{\mathbf x}_{t-1}),\qquad \frac{\partial^2\mathbf L_t}{\partial\mathbf x_{t-1}^2}=\mathbf F_t^\top\mathbf Q_t^{-1}\mathbf F_t+\mathbf P_{t-1}^{-1}=:\boldsymbol\Psi_t^{-1}.$$

$\boldsymbol\Psi_t^{-1}$ é a curvatura de $\mathbf L_t$ e faz o papel de inverso de uma covariância. Como $\mathbf L_t$ é quadrática e positiva semidefinida, o ponto onde a primeira derivada se anula é a média em $\mathbf x_{t-1}$, e a segunda derivada é a inversa da covariância:
""")

mdfig(r"""
![Figura 24 do original: L_t quadrática; a derivada nula dá a média, a curvatura dá Ψ](attachment:24.png)

Igualando a derivada a zero,

$$\mathbf x_{t-1}^\star=\boldsymbol\Psi_t\big[\mathbf F_t^\top\mathbf Q_t^{-1}(\mathbf x_t-\mathbf B_t\mathbf u_t)+\mathbf P_{t-1}^{-1}\hat{\mathbf x}_{t-1}\big],$$

e então

$$\boxed{\mathbf L_t(\mathbf x_{t-1},\mathbf x_t)=\tfrac12(\mathbf x_{t-1}-\mathbf x_{t-1}^\star)^\top\boldsymbol\Psi_t^{-1}(\mathbf x_{t-1}-\mathbf x_{t-1}^\star).}$$

(Esta é só uma das maneiras de fazer a separação.) Posto dentro da exponencial, é uma gaussiana não normalizada em $\mathbf x_{t-1}$; como a área de uma gaussiana é 1,

$$\int\exp(-\mathbf L_t(\mathbf x_{t-1},\mathbf x_t))\,d\mathbf x_{t-1}=\det(2\pi\boldsymbol\Psi_t)^{1/2},$$

que **não depende de $\mathbf x_t$**. Logo $\overline{\mathrm{bel}}(\mathbf x_t)=\eta'\exp(-\mathbf L_t(\mathbf x_t))$, e falta só calcular $\mathbf L_t(\mathbf x_t)=\mathbf L_t-\mathbf L_t(\mathbf x_{t-1},\mathbf x_t)$. Expandindo, todos os termos quadráticos e lineares em $\mathbf x_{t-1}$ se cancelam (era esse o objetivo da separação), e sobra

$$\boxed{\begin{aligned}\mathbf L_t(\mathbf x_t)=\ &\tfrac12(\mathbf x_t-\mathbf B_t\mathbf u_t)^\top\mathbf Q_t^{-1}(\mathbf x_t-\mathbf B_t\mathbf u_t)+\tfrac12\hat{\mathbf x}_{t-1}^\top\mathbf P_{t-1}^{-1}\hat{\mathbf x}_{t-1}\\&-\tfrac12\big[\mathbf F_t^\top\mathbf Q_t^{-1}(\mathbf x_t-\mathbf B_t\mathbf u_t)+\mathbf P_{t-1}^{-1}\hat{\mathbf x}_{t-1}\big]^\top\boldsymbol\Psi_t\big[\mathbf F_t^\top\mathbf Q_t^{-1}(\mathbf x_t-\mathbf B_t\mathbf u_t)+\mathbf P_{t-1}^{-1}\hat{\mathbf x}_{t-1}\big]\end{aligned}}$$

Como $\overline{\mathrm{bel}}(\mathbf x_t)$ é gaussiana, $\mathbf L_t(\mathbf x_t)$ é quadrática, e de novo a primeira derivada dá a média e a segunda a covariância. Usando o **lema de inversão de matrizes** (Woodbury), $\mathbf Q^{-1}-\mathbf Q^{-1}\mathbf F(\mathbf F^\top\mathbf Q^{-1}\mathbf F+\mathbf P^{-1})^{-1}\mathbf F^\top\mathbf Q^{-1}=(\mathbf Q+\mathbf F\mathbf P\mathbf F^\top)^{-1}$, a derivada nula dá, depois de algumas simplificações,

$$\mathbf x_t=\mathbf B_t\mathbf u_t+\mathbf F_t\hat{\mathbf x}_{t-1},\qquad \frac{\partial^2\mathbf L_t(\mathbf x_t)}{\partial\mathbf x_t^2}=(\mathbf Q_t+\mathbf F_t\mathbf P_{t-1}\mathbf F_t^\top)^{-1},$$

isto é,

$$\boxed{\hat{\mathbf x}_{t|t-1}=\mathbf F_t\hat{\mathbf x}_{t-1|t-1}+\mathbf B_t\mathbf u_t,\qquad \mathbf P_{t|t-1}=\mathbf F_t\mathbf P_{t-1|t-1}\mathbf F_t^\top+\mathbf Q_t .}$$

**Conferência numérica.** Como $\int e^{-\mathbf L_t}d\mathbf x_{t-1}=\text{const}\cdot e^{-\min_{\mathbf x_{t-1}}\mathbf L_t}$ para $\mathbf L_t$ quadrática em $\mathbf x_{t-1}$, $\mathbf L_t(\mathbf x_t)$ é, a menos de constante, o **mínimo** de $\mathbf L_t$ em $\mathbf x_{t-1}$. Conferimos os quatro ingredientes da dedução com matrizes sorteadas.
""", "24.png")

code(r"""
n, m = 3, 2
A_ = rng.normal(size=(n, n)); P_prev = A_ @ A_.T + 0.5 * np.eye(n)
A_ = rng.normal(size=(n, n)); Q_d = A_ @ A_.T + 0.3 * np.eye(n)
F_d = rng.normal(size=(n, n)); B_d = rng.normal(size=(n, 1)); u_d = rng.normal(size=1)
x_prev = rng.normal(size=n)
Qi, Pi = np.linalg.inv(Q_d), np.linalg.inv(P_prev)

def L_tot(xt, xa):
    a = xt - F_d @ xa - B_d @ u_d; b = xa - x_prev
    return 0.5 * a @ Qi @ a + 0.5 * b @ Pi @ b

Psi = np.linalg.inv(F_d.T @ Qi @ F_d + Pi)
xt0 = rng.normal(size=n)
hess_num = jacobiano_numerico(lambda xa: jacobiano_numerico(lambda y: np.array([L_tot(xt0, y)]), xa).ravel(), x_prev, eps=1e-4)
confere("∂²L/∂x²_{t-1} = Ψ⁻¹ = FᵀQ⁻¹F + P⁻¹", hess_num, np.linalg.inv(Psi), tol=1e-5, relativo=True)

def x_estrela(xt):
    return Psi @ (F_d.T @ Qi @ (xt - B_d @ u_d) + Pi @ x_prev)
grad = jacobiano_numerico(lambda xa: np.array([L_tot(xt0, xa)]), x_estrela(xt0)).ravel()
confere("x*_{t-1} anula a derivada de L_t", grad, np.zeros(n), tol=1e-6)

# L_t(x_t) = min L_t ; deve ser ½(x−m)ᵀ(Q+FPFᵀ)⁻¹(x−m) + const
m_pred = F_d @ x_prev + B_d @ u_d; P_pred = F_d @ P_prev @ F_d.T + Q_d
Lmin = lambda xt: L_tot(xt, x_estrela(xt))
pts = rng.normal(size=(6, n)) * 2
lhs = np.array([Lmin(p) - Lmin(m_pred) for p in pts])
rhs = np.array([0.5 * (p - m_pred) @ np.linalg.solve(P_pred, p - m_pred) for p in pts])
confere("L_t(x_t) − L_t(m) = ½(x−m)ᵀ(FPFᵀ+Q)⁻¹(x−m): média F x̂ + B u e covariância FPFᵀ + Q", lhs, rhs, tol=1e-9)

lhs_w = Qi - Qi @ F_d @ Psi @ F_d.T @ Qi
confere("lema de inversão: Q⁻¹ − Q⁻¹F Ψ FᵀQ⁻¹ = (Q + FPFᵀ)⁻¹", lhs_w, np.linalg.inv(Q_d + F_d @ P_prev @ F_d.T), tol=1e-9, relativo=True)
""")

md(r"""
## 8.2 Dedução da atualização (versão 1)

Queremos a média e a covariância de $\mathrm{bel}(\mathbf x_t)=\eta\,p(\mathbf z_t\mid\mathbf x_t)\,\overline{\mathrm{bel}}(\mathbf x_t)$, com $p(\mathbf z_t\mid\mathbf x_t)=\mathcal N(\mathbf H_t\mathbf x_t,\mathbf R_t)$ e $\overline{\mathrm{bel}}(\mathbf x_t)=\mathcal N(\hat{\mathbf x}_{t|t-1},\mathbf P_{t|t-1})$. Escrevendo $\mathrm{bel}(\mathbf x_t)=\eta\exp(-\mathbf J_t)$,

$$\boxed{\mathbf J_t=\tfrac12(\mathbf z_t-\mathbf H_t\mathbf x_t)^\top\mathbf R_t^{-1}(\mathbf z_t-\mathbf H_t\mathbf x_t)+\tfrac12(\mathbf x_t-\hat{\mathbf x}_{t|t-1})^\top\mathbf P_{t|t-1}^{-1}(\mathbf x_t-\hat{\mathbf x}_{t|t-1})}$$

(este $\mathbf J_t$ é uma função custo, e nada tem a ver com o jacobiano $\mathbf J$ do IESKF). Derivando,

$$\frac{\partial\mathbf J_t}{\partial\mathbf x_t}=-\mathbf H_t^\top\mathbf R_t^{-1}(\mathbf z_t-\mathbf H_t\mathbf x_t)+\mathbf P_{t|t-1}^{-1}(\mathbf x_t-\hat{\mathbf x}_{t|t-1}),\qquad \frac{\partial^2\mathbf J_t}{\partial\mathbf x_t^2}=\mathbf H_t^\top\mathbf R_t^{-1}\mathbf H_t+\mathbf P_{t|t-1}^{-1}=:\mathbf P_{t|t}^{-1}.$$

Igualando a primeira a zero e somando e subtraindo $\mathbf H_t\hat{\mathbf x}_{t|t-1}$,

$$\mathbf H_t^\top\mathbf R_t^{-1}(\mathbf z_t-\mathbf H_t\hat{\mathbf x}_{t|t-1})=(\mathbf H_t^\top\mathbf R_t^{-1}\mathbf H_t+\mathbf P_{t|t-1}^{-1})(\mathbf x_t-\hat{\mathbf x}_{t|t-1})=\mathbf P_{t|t}^{-1}(\mathbf x_t-\hat{\mathbf x}_{t|t-1}),$$

$$\therefore\quad \hat{\mathbf x}_{t|t}=\hat{\mathbf x}_{t|t-1}+\underbrace{\mathbf P_{t|t}\mathbf H_t^\top\mathbf R_t^{-1}}_{\mathbf K_t}(\mathbf z_t-\mathbf H_t\hat{\mathbf x}_{t|t-1}).$$

Isso dá a **forma de informação**: $\mathbf K_t=\mathbf P_{t|t}\mathbf H_t^\top\mathbf R_t^{-1}$, $\ \mathbf P_{t|t}=(\mathbf H_t^\top\mathbf R_t^{-1}\mathbf H_t+\mathbf P_{t|t-1}^{-1})^{-1}$. Inverter uma matriz do tamanho do estado a cada passo pode ser caro; pelo lema de inversão,

$$\mathbf P_{t|t}=\mathbf P_{t|t-1}-\mathbf P_{t|t-1}\mathbf H_t^\top(\mathbf R_t+\mathbf H_t\mathbf P_{t|t-1}\mathbf H_t^\top)^{-1}\mathbf H_t\mathbf P_{t|t-1}=(\mathbf I-\mathbf K_t\mathbf H_t)\mathbf P_{t|t-1},$$

e, multiplicando $\mathbf K_t=\mathbf P_{t|t}\mathbf H_t^\top\mathbf R_t^{-1}$ por $(\mathbf H_t\mathbf P_{t|t-1}\mathbf H_t^\top+\mathbf R_t)(\mathbf H_t\mathbf P_{t|t-1}\mathbf H_t^\top+\mathbf R_t)^{-1}=\mathbf I$ e simplificando, chega-se à forma usual

$$\boxed{\mathbf K_t=\mathbf P_{t|t-1}\mathbf H_t^\top(\mathbf H_t\mathbf P_{t|t-1}\mathbf H_t^\top+\mathbf R_t)^{-1},\qquad \hat{\mathbf x}_{t|t}=\hat{\mathbf x}_{t|t-1}+\mathbf K_t(\mathbf z_t-\mathbf H_t\hat{\mathbf x}_{t|t-1}),\qquad \mathbf P_{t|t}=(\mathbf I-\mathbf K_t\mathbf H_t)\mathbf P_{t|t-1}.}$$

> **Nota do tradutor.** No quadro final desta dedução, o original escreve "ganho de Kalman: $\mathbf K_t=\hat{\mathbf x}_{t|t-1}+\mathbf K_t(\mathbf z_t-\mathbf H_t\hat{\mathbf x}_{t|t-1})$", que é a linha da média copiada no lugar do ganho. Corrigido acima.

## 8.3 Dedução da atualização (versão 2): a gaussiana condicional

Como $\mathrm{bel}(\mathbf x_t)$ é o produto da verossimilhança pelo *a priori* (*a posteriori* = verossimilhança × *a priori*), sua média e covariância podem ser obtidas mais diretamente pela **densidade condicional**.

> **Dica (do original): a gaussiana condicional.** Se $\mathbf x$ e $\mathbf z$ são conjuntamente gaussianos, $p(\mathbf x\mid\mathbf z)=p(\mathbf x,\mathbf z)/p(\mathbf z)=\eta\,p(\mathbf z\mid\mathbf x)p(\mathbf x)\sim\mathcal N(\mathbb E(\mathbf x\mid\mathbf z),\mathbf C_{x|z})$, com
> $$\boxed{\mathbb E(\mathbf x\mid\mathbf z)=\mathbb E(\mathbf x)+\mathbf C_{xz}\mathbf C_{zz}^{-1}(\mathbf z-\mathbb E(\mathbf z)),\qquad \mathbf C_{x|z}=\mathbf C_{xx}-\mathbf C_{xz}\mathbf C_{zz}^{-1}\mathbf C_{xz}^\top.}$$

Com $\mathbf x\sim\mathcal N(\hat{\mathbf x}_{t|t-1},\mathbf P_{t|t-1})$ e $\mathbf z=\mathbf H_t\mathbf x+\mathbf v$: $\ \mathbf C_{xx}=\mathbf P_{t|t-1}$, $\ \mathbf C_{xz}=\mathbf P_{t|t-1}\mathbf H_t^\top$, $\ \mathbf C_{zz}=\mathbf H_t\mathbf P_{t|t-1}\mathbf H_t^\top+\mathbf R_t$, e $\mathbf K_t=\mathbf C_{xz}\mathbf C_{zz}^{-1}$. A figura do original resume a correspondência (lembre a troca $\mathbf Q\leftrightarrow\mathbf R$ da nota acima):
""")

mdfig(r"""
![Figura 25 do original: a correção do KF como gaussiana condicional](attachment:25.png)

**Conferência.** Abaixo, as três formas da correção (informação, usual e condicional) devem coincidir. E, como ilustração, sorteamos muitos pares $(\mathbf x,\mathbf z)$ da distribuição conjunta e fazemos a **regressão linear de $\mathbf x$ em $\mathbf z$** por mínimos quadrados: o coeficiente da regressão é o ganho de Kalman. Esta é a ponte entre o Kalman e os mínimos quadrados, que é a tese de `mq_hmm_kalman_duas_trilhas`.
""", "25.png")

code(r"""
n, m = 3, 2
A_ = rng.normal(size=(n, n)); Pp = A_ @ A_.T + 0.5 * np.eye(n)
A_ = rng.normal(size=(m, m)); Rr = A_ @ A_.T + 0.5 * np.eye(m)
Hh = rng.normal(size=(m, n)); xh = rng.normal(size=n); zz = rng.normal(size=m)

# versão 1, forma de informação
P_inf = np.linalg.inv(Hh.T @ np.linalg.inv(Rr) @ Hh + np.linalg.inv(Pp))
K_inf = P_inf @ Hh.T @ np.linalg.inv(Rr)
# versão 1, forma usual
x_us, P_us, K_us = kf_correcao(xh, Pp, zz, Hh, Rr)
# versão 2, condicional
Cxz, Czz = Pp @ Hh.T, Hh @ Pp @ Hh.T + Rr
x_cond = xh + Cxz @ np.linalg.solve(Czz, zz - Hh @ xh)
P_cond = Pp - Cxz @ np.linalg.solve(Czz, Cxz.T)
confere("ganho: P_{t|t} Hᵀ R⁻¹ = P Hᵀ (H P Hᵀ + R)⁻¹", K_inf, K_us, tol=1e-10)
confere("covariância: (HᵀR⁻¹H + P⁻¹)⁻¹ = (I − KH)P", P_inf, P_us, tol=1e-10)
confere("versão 2 (condicional) = versão 1 (média)", x_cond, x_us, tol=1e-10)
confere("versão 2 (condicional) = versão 1 (covariância)", P_cond, P_us, tol=1e-10)

# regressão de x em z, por mínimos quadrados, sobre amostras da conjunta
N_mc = 400_000
xs_mc = rng.multivariate_normal(xh, Pp, N_mc)
zs_mc = xs_mc @ Hh.T + rng.multivariate_normal(np.zeros(m), Rr, N_mc)
Zc, Xc = zs_mc - zs_mc.mean(0), xs_mc - xs_mc.mean(0)
coef = np.linalg.lstsq(Zc, Xc, rcond=None)[0].T          # x − x̄ ≈ coef (z − z̄)
confere("coeficiente da regressão de x em z = ganho de Kalman (Monte Carlo)", coef, K_us, tol=2e-2, relativo=True)
""")

# =====================================================================
md(r"""
---
# 9. MAP, Gauss–Newton e EKF

## 9.1 A dedução tradicional do EKF

Considere a observação do EKF com o ruído do lado de fora, por conveniência: $\mathbf z_t=\mathbf h(\mathbf x_t)+\mathbf v_t$. A correção do EKF desenvolve $\mathrm{bel}(\mathbf x_t)=\eta\,p(\mathbf z_t\mid\mathbf x_t)\,\overline{\mathrm{bel}}(\mathbf x_t)$, com $p(\mathbf z_t\mid\mathbf x_t)=\mathcal N(\mathbf h(\mathbf x_t),\mathbf R_t)$ e $\overline{\mathrm{bel}}(\mathbf x_t)=\mathcal N(\hat{\mathbf x}_{t|t-1},\mathbf P_{t|t-1})$, e chega a

$$\boxed{\hat{\mathbf x}_{t|t}=\hat{\mathbf x}_{t|t-1}+\mathbf K_t(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t-1})),\qquad \mathbf P_{t|t}=(\mathbf I-\mathbf K_t\mathbf H_t)\mathbf P_{t|t-1}.}$$

> **Nota do tradutor.** No original, o quadro desta subseção omite o termo $\hat{\mathbf x}_{t|t-1}+$ na média.

## 9.2 A dedução do EKF pelo MAP

**Partindo do estimador MAP** (seguindo a referência [12] do original). A correção pode ser obtida pela estimação de **máximo *a posteriori***:

$$\hat{\mathbf x}_{t|t}=\arg\max_{\mathbf x_t}\mathrm{bel}(\mathbf x_t)=\arg\max_{\mathbf x_t}p(\mathbf z_t\mid\mathbf x_t)\,\overline{\mathrm{bel}}(\mathbf x_t)=\arg\max_{\mathbf x_t}\exp\Big(-\tfrac12\big[(\mathbf z_t-\mathbf h(\mathbf x_t))^\top\mathbf R_t^{-1}(\mathbf z_t-\mathbf h(\mathbf x_t))+(\mathbf x_t-\hat{\mathbf x}_{t|t-1})^\top\mathbf P_{t|t-1}^{-1}(\mathbf x_t-\hat{\mathbf x}_{t|t-1})\big]\Big).$$

Tirando o logaritmo e o sinal negativo, o máximo vira um mínimo:

$$\boxed{\hat{\mathbf x}_{t|t}=\arg\min_{\mathbf x_t}\ \|\mathbf z_t-\mathbf h(\mathbf x_t)\|^2_{\mathbf R_t^{-1}}+\|\mathbf x_t-\hat{\mathbf x}_{t|t-1}\|^2_{\mathbf P_{t|t-1}^{-1}}}$$

com $\|\mathbf a\|^2_{\mathbf B}=\mathbf a^\top\mathbf B\mathbf a$, a norma de Mahalanobis (ao quadrado).

> **Nota do tradutor.** No original, a forma de minimização ainda aparece dentro de um $\exp(\cdot)$, e com o índice deslocado ($\mathbf P^{-1_{t|t-1}}$). Como $\exp$ é crescente, o argumento do mínimo não muda, mas o que se minimiza é o expoente, e não a exponencial; corrigido.

Em forma matricial, o custo é

$$\mathbf C_t=\begin{bmatrix}\hat{\mathbf x}_{t|t-1}-\mathbf x_t\\\mathbf z_t-\mathbf h(\mathbf x_t)\end{bmatrix}^\top\begin{bmatrix}\mathbf P_{t|t-1}^{-1}&\mathbf 0\\\mathbf 0&\mathbf R_t^{-1}\end{bmatrix}\begin{bmatrix}\hat{\mathbf x}_{t|t-1}-\mathbf x_t\\\mathbf z_t-\mathbf h(\mathbf x_t)\end{bmatrix}.$$

**Uma nova função de observação, e o MLE.** Esse custo é o de uma observação aumentada:

$$\boxed{\mathbf y_t=\mathbf g(\mathbf x_t)+\mathbf e_t,\quad \mathbf y_t=\begin{bmatrix}\hat{\mathbf x}_{t|t-1}\\\mathbf z_t\end{bmatrix},\quad \mathbf g(\mathbf x_t)=\begin{bmatrix}\mathbf x_t\\\mathbf h(\mathbf x_t)\end{bmatrix},\quad \mathbf e_t\sim\mathcal N(0,\mathbf P_{\mathbf e}),\quad \mathbf P_{\mathbf e}=\begin{bmatrix}\mathbf P_{t|t-1}&\mathbf 0\\\mathbf 0&\mathbf R_t\end{bmatrix}}$$

A predição entra como uma **medida a mais** do estado. Linearizando, $\mathbf g(\mathbf x_t)\approx\mathbf g(\hat{\mathbf x}_{t|t-1})+\mathbf J_t(\mathbf x_t-\hat{\mathbf x}_{t|t-1})$, com $\mathbf J_t=\begin{bmatrix}\mathbf I\\\mathbf H_t\end{bmatrix}$. O problema MAP de $\mathrm{bel}(\mathbf x_t)$ vira o problema de **máxima verossimilhança** (MLE) de $p(\mathbf y_t\mid\mathbf x_t)$:

$$\hat{\mathbf x}_{t|t}=\arg\min_{\mathbf x_t}\ \|\mathbf y_t-\mathbf g(\mathbf x_t)\|^2_{\mathbf P_{\mathbf e}^{-1}},$$

um problema de **mínimos quadrados ponderados** (WLS).

## 9.3 Gauss–Newton

Linearizando em $\hat{\mathbf x}_{t|t-1}$ e chamando $\delta\mathbf x=\mathbf x_t-\hat{\mathbf x}_{t|t-1}$ e $\mathbf r_t=\mathbf y_t-\mathbf g(\hat{\mathbf x}_{t|t-1})$, o problema vira $\arg\min\|\mathbf J_t\delta\mathbf x-\mathbf r_t\|^2_{\mathbf P_{\mathbf e}^{-1}}$, cuja solução são as **equações normais** de Gauss–Newton:

$$(\mathbf J_t^\top\mathbf P_{\mathbf e}^{-1}\mathbf J_t)\,\delta\mathbf x=\mathbf J_t^\top\mathbf P_{\mathbf e}^{-1}\mathbf r_t\quad\Longrightarrow\quad \boxed{\delta\mathbf x=(\mathbf J_t^\top\mathbf P_{\mathbf e}^{-1}\mathbf J_t)^{-1}\mathbf J_t^\top\mathbf P_{\mathbf e}^{-1}\mathbf r_t}$$

A matriz $\tilde{\mathbf H}=\mathbf J_t^\top\mathbf P_{\mathbf e}^{-1}\mathbf J_t$ é a **hessiana aproximada**.

**Covariância *a posteriori*.** Como $\mathbb E(\mathbf r_t\mathbf r_t^\top)=\mathbf P_{\mathbf e}$, a covariância do estimador é

$$\mathbf P_{t|t}=(\mathbf J_t^\top\mathbf P_{\mathbf e}^{-1}\mathbf J_t)^{-1}=\Big(\begin{bmatrix}\mathbf I&\mathbf H_t^\top\end{bmatrix}\begin{bmatrix}\mathbf P_{t|t-1}&\mathbf 0\\\mathbf 0&\mathbf R_t\end{bmatrix}^{-1}\begin{bmatrix}\mathbf I\\\mathbf H_t\end{bmatrix}\Big)^{-1}=(\mathbf P_{t|t-1}^{-1}+\mathbf H_t^\top\mathbf R_t^{-1}\mathbf H_t)^{-1}=(\mathbf I-\mathbf K_t\mathbf H_t)\mathbf P_{t|t-1}.$$

Ou seja, **a inversa da hessiana aproximada de Gauss–Newton é a covariância *a posteriori* do EKF**: $\tilde{\mathbf H}^{-1}=\mathbf P_{t|t}$.

> **Nota do tradutor.** O original abre essa conta com $\mathbf P_{t|t}=\mathbb E[(\mathbf x_t-\hat{\mathbf x}_{t|t-1})(\cdot)^\top]$, que é a covariância **da predição**. O que se calcula é a covariância do **estimador** $\delta\mathbf x$ em torno do valor verdadeiro, isto é, do erro *a posteriori*; o resultado final está certo.

**Média *a posteriori*.** Repetindo Gauss–Newton, a $j$-ésima iteração é

$$\begin{aligned}
\hat{\mathbf x}_{t|t,j+1}&=\hat{\mathbf x}_{t|t,j}+(\mathbf J_{t,j}^\top\mathbf P_{\mathbf e}^{-1}\mathbf J_{t,j})^{-1}\mathbf J_{t,j}^\top\mathbf P_{\mathbf e}^{-1}\big(\mathbf y_t-\mathbf g(\hat{\mathbf x}_{t|t,j})\big)\\
&=(\mathbf P_{t|t-1}^{-1}+\mathbf H_{t,j}^\top\mathbf R_t^{-1}\mathbf H_{t,j})^{-1}\big(\mathbf H_{t,j}^\top\mathbf R_t^{-1}(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j})+\mathbf H_{t,j}\hat{\mathbf x}_{t|t,j})+\mathbf P_{t|t-1}^{-1}\hat{\mathbf x}_{t|t-1}\big)\\
&=\hat{\mathbf x}_{t|t-1}+\mathbf K_{t,j}\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j})-\mathbf H_{t,j}(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j})\big).
\end{aligned}$$

A última linha é o **IEKF** (na forma correta, com a soma feita à predição). Estimar a correção do EKF por Gauss–Newton e estimá-la pelo IEKF são a mesma coisa. Na primeira iteração, $\hat{\mathbf x}_{t|t,0}=\hat{\mathbf x}_{t|t-1}$, e a fórmula se reduz a $\hat{\mathbf x}_{t|t}=\hat{\mathbf x}_{t|t-1}+\mathbf K_t(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t-1}))$, a solução do EKF. Isto é: **o EKF é Gauss–Newton com uma iteração, e o IEKF é Gauss–Newton até convergir.**

> **Nota do tradutor.** Na segunda linha dessa dedução, o original escreve o peso como $\mathbf P_{t|t-1}^{-1}$ em vez de $\mathbf P_{\mathbf e}^{-1}$. Não é só notação: $\mathbf J_t$ tem $n+m$ linhas e $\mathbf P_{t|t-1}^{-1}$ é $n\times n$, então o produto $\mathbf J_t^\top\mathbf P_{t|t-1}^{-1}\mathbf J_t$ **nem está definido**. A célula abaixo mostra isso e confere o resto com $\mathbf P_{\mathbf e}^{-1}$.
""")

code(r"""
def gauss_newton_aumentado(x_pred, P, z, h, Hfun, R, n_iter=30):
    # Gauss–Newton sobre y = [x̂_{t|t-1}; z], g(x) = [x; h(x)], peso P_e⁻¹
    n = len(x_pred)
    Pe_inv = np.linalg.inv(np.block([[P, np.zeros((n, len(z)))], [np.zeros((len(z), n)), R]]))
    y = np.concatenate([x_pred, z])
    xj = x_pred.copy(); hist = [xj.copy()]; Htil = []
    for j in range(n_iter):
        J = np.vstack([np.eye(n), Hfun(xj)])
        r = y - np.concatenate([xj, h(xj)])
        Ht = J.T @ Pe_inv @ J
        xj = xj + np.linalg.solve(Ht, J.T @ Pe_inv @ r)
        hist.append(xj.copy()); Htil.append(Ht)
    return np.array(hist), Htil

hist_gn, Htil = gauss_newton_aumentado(xp_rb, P_rb, z_rb, h_rb, H_rb, R_rb)
n_cmp = min(len(it_c), 8)
confere("iterados de Gauss–Newton no problema aumentado = iterados do IEKF correto", hist_gn[:n_cmp], it_c[:n_cmp], tol=1e-9)
x_ekf1, P_ekf1, _ = kf_correcao(xp_rb, P_rb, z_rb - h_rb(xp_rb) + H_rb(xp_rb) @ xp_rb, H_rb(xp_rb), R_rb)
confere("primeira iteração de Gauss–Newton = EKF", hist_gn[1], x_ekf1, tol=1e-12)
confere("inversa da hessiana aproximada (1ª iteração) = P_{t|t} do EKF", np.linalg.inv(Htil[0]), P_ekf1, tol=1e-12)

J0 = np.vstack([np.eye(2), H_rb(xp_rb)])
try:
    J0.T @ np.linalg.inv(P_rb) @ J0
except ValueError as e:
    print("equação impressa no original, Jᵀ P_{t|t-1}⁻¹ J  →  ValueError:", e)
""")

# =====================================================================
md(r"""
---
# 10. Dedução da correção do IESKF

Esta seção mostra as contas intermediárias que levam do problema MAP do IESKF (seção 7) à fórmula de $\delta\hat{\mathbf x}$. Simplificando os símbolos,

$$\mathbf x=\arg\min_{\mathbf x}\ \|\mathbf z-\mathbf H\mathbf x\|^2_{\mathbf R^{-1}}+\|\mathbf c+\mathbf J\mathbf x\|^2_{\mathbf P^{-1}},$$

com $\mathbf z=\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0,0)$, $\ \mathbf c=\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1}$, $\ \mathbf R=\mathbf H_{\mathbf v}\mathbf R_t\mathbf H_{\mathbf v}^\top$ (a **covariância** do ruído linearizado), $\ \mathbf P=\mathbf P_{t|t-1}$ e $\mathbf x=\delta\hat{\mathbf x}$.

**Expandindo as normas** (o termo entre colchetes, chamado $\mathbf r$ no original),

$$\mathbf r=\mathbf z^\top\mathbf R^{-1}\mathbf z-2\mathbf x^\top\mathbf H^\top\mathbf R^{-1}\mathbf z+\mathbf x^\top\mathbf H^\top\mathbf R^{-1}\mathbf H\mathbf x+\mathbf c^\top\mathbf P^{-1}\mathbf c+2\mathbf x^\top\mathbf J^\top\mathbf P^{-1}\mathbf c+\mathbf x^\top\mathbf J^\top\mathbf P^{-1}\mathbf J\mathbf x,$$

usando $\mathbf a^\top\mathbf b=\mathbf b^\top\mathbf a$. **Derivando e igualando a zero**,

$$(\mathbf H^\top\mathbf R^{-1}\mathbf H+\mathbf J^\top\mathbf P^{-1}\mathbf J)\,\mathbf x=\mathbf H^\top\mathbf R^{-1}\mathbf z-\mathbf J^\top\mathbf P^{-1}\mathbf c\quad\Longrightarrow\quad \mathbf x=(\mathbf H^\top\mathbf R^{-1}\mathbf H+\mathbf J^\top\mathbf P^{-1}\mathbf J)^{-1}(\mathbf H^\top\mathbf R^{-1}\mathbf z-\mathbf J^\top\mathbf P^{-1}\mathbf c).$$

**Lema de inversão.** Com $\bar{\mathbf P}=\mathbf J^{-1}\mathbf P\mathbf J^{-\top}$ (logo $\mathbf J^\top\mathbf P^{-1}\mathbf J=\bar{\mathbf P}^{-1}$),

$$(\bar{\mathbf P}^{-1}+\mathbf H^\top\mathbf R^{-1}\mathbf H)^{-1}=\bar{\mathbf P}-\bar{\mathbf P}\mathbf H^\top(\mathbf R+\mathbf H\bar{\mathbf P}\mathbf H^\top)^{-1}\mathbf H\bar{\mathbf P}.$$

> **Dica (do original): lema de inversão de matrizes.** $(\mathbf A+\mathbf U\mathbf C\mathbf V)^{-1}=\mathbf A^{-1}-\mathbf A^{-1}\mathbf U(\mathbf C^{-1}+\mathbf V\mathbf A^{-1}\mathbf U)^{-1}\mathbf V\mathbf A^{-1}$.

**Expandindo e eliminando termos**, usando $\mathbf H\bar{\mathbf P}\mathbf H^\top\mathbf R^{-1}=(\mathbf H\bar{\mathbf P}\mathbf H^\top+\mathbf R)\mathbf R^{-1}-\mathbf I$, os dois termos $\bar{\mathbf P}\mathbf H^\top\mathbf R^{-1}\mathbf z$ se cancelam, e sobra

$$\mathbf x=-\bar{\mathbf P}\mathbf J^\top\mathbf P^{-1}\mathbf c+\bar{\mathbf P}\mathbf H^\top\mathbf S^{-1}\mathbf z+\bar{\mathbf P}\mathbf H^\top\mathbf S^{-1}\mathbf H\bar{\mathbf P}\mathbf J^\top\mathbf P^{-1}\mathbf c,\qquad \mathbf S=\mathbf R+\mathbf H\bar{\mathbf P}\mathbf H^\top .$$

**Recuperando $\bar{\mathbf P}=\mathbf J^{-1}\mathbf P\mathbf J^{-\top}$** (então $\bar{\mathbf P}\mathbf J^\top\mathbf P^{-1}=\mathbf J^{-1}$) e **definindo $\mathbf K=\mathbf J^{-1}\mathbf P\mathbf J^{-\top}\mathbf H^\top\mathbf S^{-1}$**,

$$\mathbf x=\mathbf K\mathbf z+\mathbf K\mathbf H\mathbf J^{-1}\mathbf c-\mathbf J^{-1}\mathbf c,$$

e, **restaurando os símbolos**,

$$\boxed{\delta\hat{\mathbf x}_{t|t,j}=\mathbf K_{t,j}\big(\mathbf z_t-\mathbf h(\hat{\mathbf x}_{t|t,j},0,0)+\mathbf H_{t,j}\mathbf J_{t,j}^{-1}(\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1})\big)-\mathbf J_{t,j}^{-1}(\hat{\mathbf x}_{t|t,j}\boxminus\hat{\mathbf x}_{t|t-1}).}$$

> **Nota do tradutor.** O original define o símbolo simplificado como "$\mathbf R=\mathbf H_{\mathbf v}\mathbf R^{-1}\mathbf H_{\mathbf v}^\top$" e depois usa $\mathbf R$ como **covariância** (em $\mathbf S=\mathbf R+\mathbf H\bar{\mathbf P}\mathbf H^\top$). Só a segunda leitura é consistente; a definição foi corrigida acima (é a mesma nota da seção 7).

As duas figuras a seguir são o resumo visual do autor para esta dedução (passos [1] a [14]):
""")

mdfig(r"""
![Figura 14 do original: dedução do IESKF, passos 1 a 7](attachment:14.png)

![Figura 15 do original: dedução do IESKF, passos 8 a 14](attachment:15.png)

**Conferência.** Com matrizes sorteadas (inclusive um $\mathbf J$ qualquer, invertível), cada forma intermediária deve dar o mesmo $\mathbf x$.
""", "14.png", "15.png")

code(r"""
n, m = 4, 3
A_ = rng.normal(size=(n, n)); Pq = A_ @ A_.T + 0.5 * np.eye(n)
A_ = rng.normal(size=(m, m)); Rq = A_ @ A_.T + 0.5 * np.eye(m)
Hq = rng.normal(size=(m, n)); Jq = np.eye(n) + 0.3 * rng.normal(size=(n, n))
zq, cq = rng.normal(size=m), rng.normal(size=n)
Ri, Pinv, Ji = np.linalg.inv(Rq), np.linalg.inv(Pq), np.linalg.inv(Jq)

x_normais = np.linalg.solve(Hq.T @ Ri @ Hq + Jq.T @ Pinv @ Jq, Hq.T @ Ri @ zq - Jq.T @ Pinv @ cq)
x_otim = least_squares(lambda x: np.concatenate([np.linalg.cholesky(Ri).T @ (zq - Hq @ x),
                                                 np.linalg.cholesky(Pinv).T @ (cq + Jq @ x)]), np.zeros(n),
                       xtol=1e-15, ftol=1e-15, gtol=1e-15).x
Pbar = Ji @ Pq @ Ji.T
lema = Pbar - Pbar @ Hq.T @ np.linalg.solve(Rq + Hq @ Pbar @ Hq.T, Hq @ Pbar)
S_q = Rq + Hq @ Pbar @ Hq.T
K_q = Pbar @ Hq.T @ np.linalg.inv(S_q)
x_final = K_q @ zq + K_q @ Hq @ Ji @ cq - Ji @ cq

confere("equações normais = mínimo do problema (otimização direta)", x_normais, x_otim, tol=1e-6)
confere("lema de inversão com P̄ = J⁻¹PJ⁻ᵀ", lema, np.linalg.inv(Jq.T @ Pinv @ Jq + Hq.T @ Ri @ Hq), tol=1e-9, relativo=True)
confere("forma final: Kz + KHJ⁻¹c − J⁻¹c = solução das equações normais", x_final, x_normais, tol=1e-9)
confere("covariância: (I − KH) J⁻¹PJ⁻ᵀ = (HᵀR⁻¹H + JᵀP⁻¹J)⁻¹", (np.eye(n) - K_q @ Hq) @ Pbar,
        np.linalg.inv(Hq.T @ Ri @ Hq + Jq.T @ Pinv @ Jq), tol=1e-9, relativo=True)
""")

# =====================================================================
mdfig(r"""
---
# 11. Resumo em slides

As explicações de KF, EKF, ESKF, IEKF e IESKF podem ser resumidas, cada uma, num único slide.

### Filtro de Kalman (KF)
![Figura 09 do original: resumo do KF](attachment:09.png)

### Filtro de Kalman estendido (EKF)
![Figura 10 do original: resumo do EKF](attachment:10.png)

### Filtro de Kalman de estado de erro (ESKF)
![Figura 11 do original: resumo do ESKF](attachment:11.png)

### Filtro de Kalman estendido iterado (IEKF)
![Figura 12 do original: resumo do IEKF](attachment:12.png)

> **Nota do tradutor.** No slide do IEKF, a linha $\hat{\mathbf x}_{t|t,j+1}=\hat{\mathbf x}_{t|t,j}+\delta\hat{\mathbf x}_{t|t,j}$, com $\delta\hat{\mathbf x}_{t|t,j}=\mathbf K_{t,j}(\mathbf z_t-\mathbf h(\cdot)-\mathbf H_t(\hat{\mathbf x}_{t|t-1}-\hat{\mathbf x}_{t|t,j}))$, tem o erro discutido na seção 6: a soma deve ser feita à predição $\hat{\mathbf x}_{t|t-1}$.

### Filtro de Kalman de estado de erro iterado (IESKF)
![Figura 13 do original: resumo do IESKF](attachment:13.png)
""", "09.png", "10.png", "11.png", "12.png", "13.png")

mdfig(r"""
### Figuras do código-fonte que o texto não usa

O arquivo do arXiv traz mais quatro figuras, que não aparecem no PDF. Duas são versões antigas dos slides de KF e EKF, com a convenção de *Probabilistic Robotics* ($\mathbf R$ para o movimento, $\mathbf Q$ para a medida, e o ruído $\mathbf F_{\mathbf w}\mathbf Q\mathbf F_{\mathbf w}^\top$ ainda ausente); a outra põe EKF e ESKF lado a lado, e é uma boa síntese da seção 5.

![Figura 01 do código-fonte: EKF e ESKF lado a lado](attachment:01.png)

![Figura 04 do código-fonte: versão antiga do slide do KF (R e Q trocados)](attachment:04.png) ![Figura 02 do código-fonte: versão antiga do slide do EKF (R e Q trocados)](attachment:02.png)
""", "01.png", "04.png", "02.png")

md(r"""
### Acréscimo: a família numa tabela

| filtro | estima | jacobianos em relação a | correção | onde aparece na série |
|---|---|---|---|---|
| **KF** | $\mathbf x$ | (modelo linear) | uma vez | `kalman_passo_a_passo`, Faragher, `mq_hmm_kalman_duas_trilhas` |
| **EKF** | $\mathbf x$ | $\mathbf x$ (e ruídos: $\mathbf F_{\mathbf w},\mathbf H_{\mathbf v}$) | uma vez = 1 passo de Gauss–Newton | `pepe_kalman_parte1_ekf_iekf` |
| **ESKF** | $\delta\mathbf x$ em torno do nominal | $\delta\mathbf x$ | uma vez, depois injeção $\boxplus$ e reset $\mathbf G$ | este notebook (atitude) |
| **IEKF** | $\mathbf x$ | $\mathbf x$, relinearizados | até convergir = Gauss–Newton = MAP | `pepe_kalman_parte1_ekf_iekf`, seção C5b de `mq_hmm_kalman_duas_trilhas` |
| **IESKF** | $\delta\mathbf x$ em torno do iterado | $\delta\mathbf x$, relinearizados, mais $\mathbf J$ | até convergir = Gauss–Newton na variedade | este notebook (atitude) |

Uma regra para guardar: **"estendido" = linearizar; "iterado" = Gauss–Newton até convergir; "de estado de erro" = trabalhar num espaço vetorial tangente quando o estado não é um vetor.** As três ideias são independentes e se combinam.
""")

# =====================================================================
md(r"""
---
# 12. Notas do tradutor: lista consolidada

Todas as correções feitas no texto, na ordem em que aparecem. A coluna "gravidade" separa os erros que **mudam o resultado** dos que são só de digitação.

| # | onde (no original) | o que está escrito | correção | gravidade | conferido em |
|---|---|---|---|---|---|
| 1 | §2, dedução do filtro de Bayes | última linha com $\overline{\mathrm{bel}}(\mathbf x_{t-1})$ | $\overline{\mathrm{bel}}(\mathbf x_t)$ | digitação | — |
| 2 | §3 e §4, valor inicial | "$\mathbf P_0$ tipicamente pequeno ($<10^{-2}$)" com $\hat{\mathbf x}_0=0$ | $\mathbf P_0$ deve refletir a incerteza inicial: chute ⇒ $\mathbf P_0$ grande | **muda o resultado** | §3 (NEES no transitório) |
| 3 | §3.1, KF 1D | $x_t=x_{t-1}+u_t+\sigma^2_{\text{motion}}$ | soma-se um ruído de variância $\sigma^2$ | conceitual | — |
| 4 | §3.2, ganho | "$\mathbf K_t\to\mathbf 1$" | só vale no caso escalar com $H=1$ | conceitual | — |
| 5 | §5, ESKF | "o EKF é lento e acumula erro; o ESKF é rápido e preciso" | só há diferença quando o estado não é vetor | conceitual | §5.3 (ESKF = EKF) |
| 6 | §5.1, modelos do ESKF | o termo $\mathbf H_t\delta\mathbf x_t$ some da observação | o erro estimado é zero, o erro não | conceitual | — |
| 7 | §5.2, correção do ESKF | $\mathbf H_{\mathbf v}\mathbf R_t\mathbf H_{\mathbf v}$ | $\mathbf H_{\mathbf v}\mathbf R_t\mathbf H_{\mathbf v}^\top$ | digitação | — |
| 8 | §5.2, reset | $\mathbf g(\delta\mathbf x)=\delta\mathbf x-\delta\hat{\mathbf x}_{t\vert t-1}$ | desconta-se $\delta\hat{\mathbf x}_{t\vert t}$ (com $\boxminus$) | conceitual | §5.4 ($\mathbf G=\mathbf J_r$) |
| 9 | §5.3, resumo do ESKF | $\mathbf F_t\hat{\mathbf x}_{t-1\vert t-1}$; $\mathbf G\mathbf P\mathbf P^\top$; retorna $\delta\hat{\mathbf x}$ | $\mathbf F_t\delta\hat{\mathbf x}_{t-1\vert t-1}$; $\mathbf G\mathbf P\mathbf G^\top$; retorna $\hat{\mathbf x}_{t\vert t}$ | digitação | — |
| 10 | **§6, correção do IEKF (texto, resumo, figs. 07 e 12)** | $\hat{\mathbf x}_{t\vert t,j+1}=\hat{\mathbf x}_{t\vert t,j}+\mathbf K\mathbf r_j$ | $\hat{\mathbf x}_{t\vert t,j+1}=\hat{\mathbf x}_{t\vert t-1}+\mathbf K\mathbf r_j$ | **diverge** | §6 (caso linear e não linear) |
| 11 | §7, pesos do IESKF; §10, símbolo $\mathbf R$ | $\mathbf H_{\mathbf v}\mathbf R^{-1}\mathbf H_{\mathbf v}^\top$ | $(\mathbf H_{\mathbf v}\mathbf R\mathbf H_{\mathbf v}^\top)^{-1}$ | só coincide se $\mathbf H_{\mathbf v}$ ortogonal | §10 |
| 12 | §7, problema MAP do IESKF | $\mathbf h(\hat{\mathbf x}_{t\vert t-1},0,0)$ | $\mathbf h(\hat{\mathbf x}_{t\vert t,j},0,0)$ | conceitual | §7 (atitude) |
| 13 | §7, resumo do IESKF | $\mathbf F_t\hat{\mathbf x}_{t-1\vert t-1}$ | $\mathbf F_t\delta\hat{\mathbf x}_{t-1\vert t-1}$ | digitação | — |
| 14 | §8 inteira e fig. 25 | $\mathbf R$ = ruído de movimento, $\mathbf Q$ = ruído de medida | convenção trocada em relação ao resto do texto | confusão de notação | — |
| 15 | §8.2, quadro final | "$\mathbf K_t=\hat{\mathbf x}_{t\vert t-1}+\mathbf K_t(\cdot)$" | $\mathbf K_t=\mathbf P_{t\vert t-1}\mathbf H_t^\top(\mathbf H_t\mathbf P_{t\vert t-1}\mathbf H_t^\top+\mathbf R_t)^{-1}$ | digitação | §8.3 |
| 16 | §9.1, EKF tradicional | $\hat{\mathbf x}_{t\vert t}=\mathbf K_t(\cdot)$ | $\hat{\mathbf x}_{t\vert t}=\hat{\mathbf x}_{t\vert t-1}+\mathbf K_t(\cdot)$ | digitação | — |
| 17 | §9.2, MAP | $\arg\min\exp(\cdot)$ e $\mathbf P^{-1_{t\vert t-1}}$ | minimiza-se o expoente | digitação | — |
| 18 | §9.3, covariância *a posteriori* | $\mathbb E[(\mathbf x_t-\hat{\mathbf x}_{t\vert t-1})(\cdot)^\top]$ | covariância do erro do estimador | conceitual (resultado certo) | §9 |
| 19 | §9.3, iteração de GN | peso $\mathbf P_{t\vert t-1}^{-1}$ | $\mathbf P_{\mathbf e}^{-1}$ (dimensões incompatíveis no original) | **produto indefinido** | §9 (`ValueError`) |

**Leitura.** São erros de anotações de estudo, não de um livro revisado, e quase todos são de digitação. O único que **muda o algoritmo** é o nº 10, e ele é desmentido pelo próprio texto em dois lugares (seção 9, última linha; seção 7, com $\mathbf J=\mathbf I$). Encontrar essa contradição lendo o texto com atenção, e confirmá-la com dez linhas de código, é talvez a lição mais útil deste notebook.
""")

md(r"""
---
# 13. Exercícios

1. **$\mathbf P_0$ e confiança.** Na seção 3, encontre o menor $\mathbf P_0=p\,\mathbf I$ para o qual o NEES médio dos passos 1–20 fica abaixo de 6. Relacione $p$ com o erro inicial ($50$ m na posição, $4$ m/s na velocidade).

2. **O EKF quebra.** No exemplo da seção 6, aumente $P_{yy}$ da predição de 4 para 16 e depois para 64. Para cada caso, compare o erro do EKF (iteração 1) e do IEKF até o MAP. A partir de que incerteza o EKF fica pior que a própria predição?

3. **Bell e Cathey em uma linha.** Mostre algebricamente que a forma "incremental" $\hat{\mathbf x}_{j+1}=\hat{\mathbf x}_j+\big[\mathbf K_j\mathbf r_j-(\hat{\mathbf x}_j-\hat{\mathbf x}_{t|t-1})\big]$ coincide com a forma da seção 6.3, e implemente-a numa linha alterando `iekf_correcao`.

4. **Observabilidade na atitude.** Na seção 5.4, retire o magnetômetro (use só as três primeiras componentes de $\mathbf z$ e de $\mathbf H$). Rode o ESKF e explique o que acontece com o desvio reportado de $\delta\theta_z$ e com o erro verdadeiro. Por que o acelerômetro não "vê" a rotação em torno da vertical?

5. **Quaternions em vez de matrizes.** Reescreva `boxplus` e `boxminus` usando quaternions unitários, com $\mathbf q\boxplus\delta\boldsymbol\theta=\mathbf q\otimes\mathrm{Exp}_q(\delta\boldsymbol\theta)$. Confira que o ESKF da seção 5.4 dá exatamente as mesmas estimativas.

6. **O EKF "ingênuo" no quaternion.** Implemente um EKF cujo estado é o quaternion $\mathbf q\in\mathbb R^4$ (covariância $4\times4$), somando a correção diretamente e normalizando depois. Compare com o ESKF no mesmo cenário: erro, NEES e o que acontece com a covariância na direção radial $\mathbf q$.

7. **Gauss–Newton amortecido.** Troque, em `gauss_newton_aumentado`, o passo de Gauss–Newton por um passo de Levenberg–Marquardt, $(\tilde{\mathbf H}+\lambda\mathbf I)\delta\mathbf x=\mathbf J^\top\mathbf P_{\mathbf e}^{-1}\mathbf r$. Em que situação isso ajuda o IEKF? (Dica: medidas muito precisas de uma função muito curva, longe da predição.)
""")

md(r"""
---
# 14. Referências

**Obra traduzida.** G. E. Im, *Notes on Kalman Filter (KF, EKF, ESKF, IEKF, IESKF)*, arXiv:2406.06427v3, 2024. Licença CC BY (conforme informado). Histórico de revisões do original: 13 versões, de 23/06/2020 a 28/06/2024.

**Referências do original** (numeração mantida, citadas no texto como [n]):
1. [Kalman filter — Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter).
2. J. Solà, *Quaternion kinematics for the error-state Kalman filter*, [arXiv:1711.02508](https://arxiv.org/abs/1711.02508), 2017.
3. *Robot Mapping*, curso da Universidade de Freiburg (C. Stachniss), [vídeo](https://youtu.be/wVsfCnyt5jA).
4. J. Jeong, [*[SLAM] Kalman filter and EKF*](http://jinyongjeong.github.io/2017/02/14/lec03_kalman_filter_and_EKF/) (blog).
5. [*Error-State Kalman Filter: compreensão e dedução das fórmulas*](https://blog.csdn.net/liu3612162/article/details/114120772) (CSDN, blog).
6. D. He, W. Xu, F. Zhang, *Kalman filters on differentiable manifolds*, [arXiv:2102.03804](https://arxiv.org/abs/2102.03804), 2021.
7. X. Gao et al., [*SLAM in Autonomous Driving*](https://github.com/gaoxiang12/slam_in_autonomous_driving) (livro e código).
8. W. Xu, F. Zhang, *FAST-LIO: a fast, robust LiDAR-inertial odometry package by tightly-coupled iterated Kalman filter*, IEEE RA-L 6(2):3317–3324, 2021, [arXiv:2010.08196](https://arxiv.org/abs/2010.08196).
9. J. Huai, X. Gao, *A quick guide for the iterated extended Kalman filter on manifolds*, [arXiv:2307.09237](https://arxiv.org/abs/2307.09237), 2023.
10. M. Bloesch et al., *Iterated extended Kalman filter based visual-inertial odometry using direct photometric feedback*, IJRR 36(10):1053–1072, 2017.
11. M. A. Skoglund, G. Hendeby, D. Axehill, *Extended Kalman filter modifications based on an optimization view point*, FUSION 2015.
12. [*From MAP, MLE, OLS, GN to IEKF, EKF*](https://cgabc.xyz/posts/784a80cb/) (blog).
13. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005 (o original cita, por engano, o artigo homônimo de Thrun em *Communications of the ACM* 45(3), 2002).

**Acrescentadas nesta tradução.**
- B. M. Bell, F. W. Cathey, *The iterated Kalman filter update as a Gauss–Newton method*, IEEE Trans. Automatic Control 38(2):294–297, 1993. (A forma correta do IEKF.)
- V. Madyastha et al., *Extended Kalman filter vs. error state Kalman filter for aircraft attitude estimation*, AIAA GNC, 2011. (Citado no original sem entrada na lista.)
- Os outros notebooks da série; veja `kalman_guia_da_serie`.
""")
